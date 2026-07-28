"""Offline-only low-capacity Pairwise Selector learning curves."""

import argparse
import collections
import csv
import datetime
import json
import os
import subprocess
from pathlib import Path

import jax
import numpy as np

from diagnostics.low_capacity_selector import (
    MODEL_NAMES,
    PARAMETER_COUNTS,
    build_nested_episode_subsets,
    create_low_capacity_state,
    low_capacity_evaluate_objective,
    low_capacity_predict,
    low_capacity_train_step,
    margin_bucket_accuracy,
    normalized_state_inputs,
)
from diagnostics.pairwise_selector import (
    PAIR_INDICES,
    compute_normalization,
    load_and_validate_dataset,
    ranking_metrics,
    restore_state,
    save_state,
    sha256_file,
)


DEFAULT_DATASET = "/root/autodl-tmp/qc_workspace/datasets/qc_branch_preferences_h80_seed0.npz"
DEFAULT_OUTPUT = "/root/autodl-tmp/qc_workspace/experiments/pairwise_selector_low_capacity_seed0"
V1_METRICS = "/root/autodl-tmp/qc_workspace/experiments/pairwise_selector_v1_seed0/metrics.json"
TARGET_SIZES = (25, 50, 100, 140)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def aggregate_epoch(batch_metrics):
    counts = np.asarray(
        [values["pair_count"] for values in batch_metrics], dtype=np.float64
    )
    total = max(float(counts.sum()), 1.0)
    result = {}
    for key in ("loss", "pairwise_accuracy"):
        result[key] = float(
            sum(
                values[key] * count
                for values, count in zip(batch_metrics, counts)
            )
            / total
        )
    for source, target in (
        ("grad_norm", "gradient_norm"),
        ("score_mean", "score_mean"),
        ("score_std", "score_std"),
    ):
        result[target] = float(
            np.mean([values[source] for values in batch_metrics])
        )
    result["pair_count"] = int(counts.sum())
    result["gradient_finite"] = bool(
        all(values["gradient_finite"] for values in batch_metrics)
    )
    return result


def subset_coverage(arrays, subset):
    first_indices = []
    for state_id in subset.state_ids:
        indices = np.flatnonzero(arrays["state_id"] == state_id)
        first_indices.append(indices[np.argmin(arrays["candidate_index"][indices])])
    first_indices = np.asarray(first_indices)
    episode_ids = arrays["episode_id"][first_indices]
    episode_success = {}
    for episode_id, success in zip(
        episode_ids, arrays["baseline_episode_success"][first_indices]
    ):
        episode_success[int(episode_id)] = bool(success)
    deltas = np.abs(
        subset.returns[:, PAIR_INDICES[:, 0]]
        - subset.returns[:, PAIR_INDICES[:, 1]]
    )
    return {
        "states": len(subset.state_ids),
        "episodes": len(np.unique(episode_ids)),
        "success_episodes": sum(episode_success.values()),
        "failure_episodes": len(episode_success) - sum(episode_success.values()),
        "progress_distribution": dict(
            collections.Counter(
                str(int(value))
                for value in arrays["initial_progress"][first_indices]
            )
        ),
        "phase_distribution": dict(
            collections.Counter(
                str(value) for value in arrays["state_phase"][first_indices]
            )
        ),
        "epsilon_1_pair_count": int(np.sum(deltas >= 1.0)),
        "tied_states": int(np.sum(np.ptp(subset.returns, axis=1) == 0)),
    }


def subgroup_metrics(split, scores):
    result = {}
    for name, mask in (
        ("overall", np.ones(len(split.state_ids), dtype=bool)),
        ("success_episode", split.baseline_episode_success),
        ("failure_episode", ~split.baseline_episode_success),
    ):
        if not np.any(mask):
            result[name] = None
            continue
        result[name] = ranking_metrics(split.returns[mask], scores[mask])
        result[name]["episodes"] = int(
            len(np.unique(split.episode_ids[mask]))
        )
    return result


def train_one(
    model_name,
    target_size,
    train_subset,
    validation,
    test,
    normalization,
    run_dir,
    args,
):
    inputs = {
        "train": normalized_state_inputs(train_subset, normalization),
        "validation": normalized_state_inputs(validation, normalization),
        "test": normalized_state_inputs(test, normalization),
    }
    state = create_low_capacity_state(
        model_name,
        args.seed,
        args.learning_rate,
        args.weight_decay,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "normalization.npz",
        observation_mean=normalization.observation_mean,
        observation_std=normalization.observation_std,
        action_mean=normalization.action_mean,
        action_std=normalization.action_std,
        observation_near_zero=normalization.observation_near_zero,
        action_near_zero=normalization.action_near_zero,
    )
    history = []
    best_loss = float("inf")
    best_epoch = 0
    best_validation_predictions = None
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        permutation = np.random.default_rng(args.seed + epoch).permutation(
            len(train_subset.state_ids)
        )
        batch_metrics = []
        for start in range(0, len(permutation), args.batch_states):
            indices = permutation[start : start + args.batch_states]
            state, metrics = low_capacity_train_step(
                state,
                inputs["train"][0][indices],
                inputs["train"][1][indices],
                train_subset.returns[indices],
                np.float32(args.pair_epsilon),
            )
            metrics = {
                key: np.asarray(value).item() for key, value in metrics.items()
            }
            if not metrics["gradient_finite"] or not np.isfinite(
                metrics["loss"]
            ):
                raise FloatingPointError(
                    f"Non-finite {model_name} target={target_size} epoch={epoch}"
                )
            batch_metrics.append(metrics)
        train_metrics = aggregate_epoch(batch_metrics)
        validation_metrics = {
            key: np.asarray(value).item()
            for key, value in low_capacity_evaluate_objective(
                state,
                *inputs["validation"],
                validation.returns,
                np.float32(args.pair_epsilon),
            ).items()
        }
        record = {
            "epoch": epoch,
            "step": int(state.step),
            "train_pairwise_loss": train_metrics["loss"],
            "validation_pairwise_loss": float(validation_metrics["loss"]),
            "train_pairwise_accuracy": train_metrics["pairwise_accuracy"],
            "validation_pairwise_accuracy": float(
                validation_metrics["pairwise_accuracy"]
            ),
            "gradient_norm": train_metrics["gradient_norm"],
            "gradient_finite": train_metrics["gradient_finite"],
            "train_score_mean": train_metrics["score_mean"],
            "train_score_std": train_metrics["score_std"],
            "train_validation_loss_gap": float(
                validation_metrics["loss"] - train_metrics["loss"]
            ),
        }
        history.append(record)
        if record["validation_pairwise_loss"] < best_loss - 1e-6:
            best_loss = record["validation_pairwise_loss"]
            best_epoch = epoch
            stale = 0
            save_state(run_dir / "best.pkl", state)
            best_validation_predictions = np.asarray(
                low_capacity_predict(state, *inputs["validation"])
            )
            restored = restore_state(
                run_dir / "best.pkl",
                create_low_capacity_state(
                    model_name,
                    args.seed,
                    args.learning_rate,
                    args.weight_decay,
                ),
            )
            np.testing.assert_array_equal(
                best_validation_predictions,
                np.asarray(
                    low_capacity_predict(restored, *inputs["validation"])
                ),
            )
        else:
            stale += 1
        if epoch == 1 or epoch % 50 == 0:
            print(
                f"model={model_name} target={target_size} actual={len(train_subset.state_ids)} "
                f"epoch={epoch} train_loss={record['train_pairwise_loss']:.6f} "
                f"val_loss={record['validation_pairwise_loss']:.6f} "
                f"train_acc={record['train_pairwise_accuracy']:.4f} "
                f"val_acc={record['validation_pairwise_accuracy']:.4f}",
                flush=True,
            )
        if stale >= args.patience:
            break
    save_state(run_dir / "latest.pkl", state)
    best_state = restore_state(
        run_dir / "best.pkl",
        create_low_capacity_state(
            model_name,
            args.seed,
            args.learning_rate,
            args.weight_decay,
        ),
    )
    predictions = {}
    evaluations = {}
    for split_name, split in (
        ("train", train_subset),
        ("validation", validation),
        ("test", test),
    ):
        predictions[split_name] = np.asarray(
            low_capacity_predict(best_state, *inputs[split_name])
        )
        evaluations[split_name] = ranking_metrics(
            split.returns, predictions[split_name]
        )
    with open(run_dir / "history.json", "w") as file:
        json.dump(json_ready(history), file, indent=2)
    with open(run_dir / "history.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return {
        "model": model_name,
        "target_states": target_size,
        "actual_train_states": len(train_subset.state_ids),
        "parameter_count": PARAMETER_COUNTS[model_name],
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_step": history[best_epoch - 1]["step"],
        "best_validation_loss": best_loss,
        "best_training_record": history[best_epoch - 1],
        "final_training_record": history[-1],
        "normalization": {
            "observation_mean_shape": list(normalization.observation_mean.shape),
            "observation_std_shape": list(normalization.observation_std.shape),
            "action_mean_shape": list(normalization.action_mean.shape),
            "action_std_shape": list(normalization.action_std.shape),
            "observation_std_min": float(normalization.observation_std.min()),
            "action_std_min": float(normalization.action_std.min()),
            "observation_near_zero": normalization.observation_near_zero,
            "action_near_zero": normalization.action_near_zero,
            "std_floor": 1e-6,
        },
        "evaluations": evaluations,
        "margin_buckets": {
            split_name: margin_bucket_accuracy(split.returns, predictions[split_name])
            for split_name, split in (
                ("validation", validation),
                ("test", test),
            )
        },
        "test_subgroups": subgroup_metrics(test, predictions["test"]),
        "best_checkpoint": str(run_dir / "best.pkl"),
        "latest_checkpoint": str(run_dir / "latest.pkl"),
        "history_json": str(run_dir / "history.json"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pair_epsilon", type=float, default=1.0)
    parser.add_argument("--batch_states", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    args = parser.parse_args()

    np.random.seed(args.seed)
    arrays, grouped = load_and_validate_dataset(args.dataset)
    subsets, subset_metadata = build_nested_episode_subsets(
        grouped["train"], args.seed
    )
    validation_ids = np.array(grouped["validation"].state_ids, copy=True)
    test_ids = np.array(grouped["test"].state_ids, copy=True)
    subset_coverage_metadata = {}
    for target, subset in subsets.items():
        subset_coverage_metadata[target] = {
            **subset_metadata[target],
            **subset_coverage(arrays, subset),
        }
        np.testing.assert_array_equal(
            validation_ids, grouped["validation"].state_ids
        )
        np.testing.assert_array_equal(test_ids, grouped["test"].state_ids)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "metadata_sha256": sha256_file(
            str(Path(args.dataset).with_suffix(".json"))
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": args.seed,
        "subset_seed": args.seed,
        "pair_epsilon": args.pair_epsilon,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_states": args.batch_states,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "models": list(MODEL_NAMES),
        "parameter_counts": PARAMETER_COUNTS,
        "normalization": "each training subset only; std floor 1e-6",
        "training_timestamp_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
    }
    with open(output_dir / "config.json", "w") as file:
        json.dump(json_ready(config), file, indent=2, sort_keys=True)
    print(f"subset_metadata={json.dumps(json_ready(subset_coverage_metadata), sort_keys=True)}")

    results = {model_name: {} for model_name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for target in TARGET_SIZES:
            subset = subsets[target]
            normalization = compute_normalization(subset)
            run_dir = output_dir / model_name / f"n{target}"
            result = train_one(
                model_name,
                target,
                subset,
                grouped["validation"],
                grouped["test"],
                normalization,
                run_dir,
                args,
            )
            results[model_name][target] = result
            print(
                f"completed model={model_name} target={target} "
                f"actual={result['actual_train_states']} best_epoch={result['best_epoch']} "
                f"val_pairwise={result['evaluations']['validation']['pairwise_accuracy']:.6f} "
                f"test_pairwise={result['evaluations']['test']['pairwise_accuracy']:.6f}",
                flush=True,
            )

    critic = {
        split: ranking_metrics(values.returns, values.critic_scores)
        for split, values in grouped.items()
    }
    critic["test_subgroups"] = subgroup_metrics(
        grouped["test"], grouped["test"].critic_scores
    )
    n140_candidates = [results[model][140] for model in MODEL_NAMES]
    validation_selected = max(
        n140_candidates,
        key=lambda result: (
            result["evaluations"]["validation"]["pairwise_accuracy"],
            -result["best_validation_loss"],
        ),
    )
    with open(V1_METRICS) as file:
        v1_metrics = json.load(file)
    summary = {
        "config": config,
        "subset_metadata": subset_coverage_metadata,
        "results": results,
        "critic": critic,
        "v1_reference": {
            "full": v1_metrics["variants"]["full"]["evaluations"],
            "action_only": v1_metrics["variants"]["action_only"]["evaluations"],
            "v1_parameter_count": 25857,
        },
        "validation_selected_n140_model": validation_selected["model"],
        "validation_selection_rule": (
            "highest n140 validation pairwise accuracy, then lowest validation loss"
        ),
    }
    with open(output_dir / "metrics.json", "w") as file:
        json.dump(json_ready(summary), file, indent=2, sort_keys=True)
    print(f"critic={json.dumps(json_ready(critic), sort_keys=True)}")
    print(
        f"validation_selected_n140_model={validation_selected['model']}"
    )
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
