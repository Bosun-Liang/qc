"""Train and evaluate the offline-only Pairwise Selector v1 experiment."""

import argparse
import csv
import dataclasses
import datetime
import json
import os
import subprocess
from pathlib import Path

import jax
import numpy as np

from diagnostics.pairwise_selector import (
    INPUT_DIMS,
    NUM_CANDIDATES,
    compute_normalization,
    count_parameters,
    create_train_state,
    evaluate_objective,
    load_and_validate_dataset,
    normalized_inputs,
    pair_statistics,
    predict_scores,
    ranking_metrics,
    restore_state,
    save_state,
    sha256_file,
    train_step,
)


DEFAULT_DATASET = "/root/autodl-tmp/qc_workspace/datasets/qc_branch_preferences_h80_seed0.npz"
DEFAULT_OUTPUT = "/root/autodl-tmp/qc_workspace/experiments/pairwise_selector_v1_seed0"
VARIANTS = ("full", "shuffle", "observation_only", "action_only")


def json_ready(value):
    if dataclasses.is_dataclass(value):
        return json_ready(dataclasses.asdict(value))
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
    pair_counts = np.asarray([values["pair_count"] for values in batch_metrics], dtype=np.float64)
    total_pairs = max(float(pair_counts.sum()), 1.0)
    result = {}
    for key in ("loss", "pairwise_accuracy"):
        result[key] = float(
            np.sum([values[key] * count for values, count in zip(batch_metrics, pair_counts)]) / total_pairs
        )
    for source, target in (
        ("grad_norm", "gradient_norm"),
        ("score_mean", "score_mean"),
        ("score_std", "score_std"),
    ):
        result[target] = float(
            np.mean([values[source] for values in batch_metrics])
        )
    result["pair_count"] = int(pair_counts.sum())
    result["gradient_finite"] = bool(all(values["gradient_finite"] for values in batch_metrics))
    return result


def shuffled_train_returns(returns, seed):
    """Break return-to-candidate correspondence independently within states."""
    rng = np.random.default_rng(seed + 2718)
    shuffled = np.array(returns, copy=True)
    for state_index in range(len(shuffled)):
        shuffled[state_index] = shuffled[state_index, rng.permutation(NUM_CANDIDATES)]
    if not np.array_equal(np.sort(shuffled, axis=1), np.sort(returns, axis=1)):
        raise AssertionError("Shuffle control changed per-state label multisets")
    return shuffled


def train_variant(
    variant,
    grouped,
    normalization,
    output_dir,
    seed,
    learning_rate,
    weight_decay,
    epsilon,
    batch_states,
    max_epochs,
    patience,
):
    model_mode = "full" if variant == "shuffle" else variant
    inputs = {
        split: normalized_inputs(values, normalization, model_mode)
        for split, values in grouped.items()
    }
    train_returns = grouped["train"].returns
    if variant == "shuffle":
        train_returns = shuffled_train_returns(train_returns, seed)
    state = create_train_state(model_mode, seed, learning_rate, weight_decay)
    initial_state = create_train_state(model_mode, seed, learning_rate, weight_decay)
    for left, right in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(initial_state.params)):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    variant_dir = Path(output_dir) / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_prediction = None

    for epoch in range(1, max_epochs + 1):
        permutation = np.random.default_rng(seed + epoch).permutation(len(train_returns))
        batches = []
        for start in range(0, len(permutation), batch_states):
            indices = permutation[start : start + batch_states]
            state, batch = train_step(
                state,
                inputs["train"][indices],
                train_returns[indices],
                np.float32(epsilon),
            )
            batch = {key: np.asarray(value).item() for key, value in batch.items()}
            if not batch["gradient_finite"] or not np.isfinite(batch["loss"]):
                raise FloatingPointError(f"Non-finite update in {variant} epoch {epoch}")
            batches.append(batch)
        train_metrics = aggregate_epoch(batches)
        validation_metrics = {
            key: np.asarray(value).item()
            for key, value in evaluate_objective(
                state,
                inputs["validation"],
                grouped["validation"].returns,
                np.float32(epsilon),
            ).items()
        }
        record = {
            "epoch": epoch,
            "step": int(state.step),
            "train_pairwise_loss": train_metrics["loss"],
            "validation_pairwise_loss": float(validation_metrics["loss"]),
            "train_pairwise_accuracy": train_metrics["pairwise_accuracy"],
            "validation_pairwise_accuracy": float(validation_metrics["pairwise_accuracy"]),
            "gradient_norm": train_metrics["gradient_norm"],
            "gradient_finite": train_metrics["gradient_finite"],
            "train_score_mean": train_metrics["score_mean"],
            "train_score_std": train_metrics["score_std"],
            "train_validation_loss_gap": float(validation_metrics["loss"] - train_metrics["loss"]),
        }
        history.append(record)
        if record["validation_pairwise_loss"] < best_loss - 1e-6:
            best_loss = record["validation_pairwise_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_state(variant_dir / "best.pkl", state)
            best_prediction = np.asarray(predict_scores(state, inputs["validation"]))
            restored = restore_state(variant_dir / "best.pkl", create_train_state(model_mode, seed, learning_rate, weight_decay))
            np.testing.assert_allclose(
                np.asarray(predict_scores(restored, inputs["validation"])),
                best_prediction,
                rtol=0,
                atol=0,
            )
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 25 == 0:
            print(
                f"variant={variant} epoch={epoch} train_loss={record['train_pairwise_loss']:.6f} "
                f"val_loss={record['validation_pairwise_loss']:.6f} "
                f"train_acc={record['train_pairwise_accuracy']:.4f} "
                f"val_acc={record['validation_pairwise_accuracy']:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= patience:
            break

    save_state(variant_dir / "latest.pkl", state)
    best_state = restore_state(
        variant_dir / "best.pkl",
        create_train_state(model_mode, seed, learning_rate, weight_decay),
    )
    if best_prediction is not None:
        np.testing.assert_allclose(
            np.asarray(predict_scores(best_state, inputs["validation"])),
            best_prediction,
            rtol=0,
            atol=0,
        )
    evaluations = {}
    predictions = {}
    for split in ("train", "validation", "test"):
        predictions[split] = np.asarray(predict_scores(best_state, inputs[split]))
        evaluations[split] = ranking_metrics(grouped[split].returns, predictions[split])
    with open(variant_dir / "history.json", "w") as file:
        json.dump(json_ready(history), file, indent=2)
    with open(variant_dir / "history.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    return {
        "variant": variant,
        "input_mode": model_mode,
        "input_dim": INPUT_DIMS[model_mode],
        "parameter_count": count_parameters(best_state.params),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_step": int(history[best_epoch - 1]["step"]),
        "best_validation_loss": best_loss,
        "best_training_record": history[best_epoch - 1],
        "final_training_record": history[-1],
        "evaluations": evaluations,
        "history_path": str(variant_dir / "history.json"),
        "best_checkpoint": str(variant_dir / "best.pkl"),
        "latest_checkpoint": str(variant_dir / "latest.pkl"),
    }, predictions


def subgroup_metrics(test_split, scores):
    result = {}
    for name, mask in (
        ("overall", np.ones(len(test_split.state_ids), dtype=bool)),
        ("success_episode", test_split.baseline_episode_success),
        ("failure_episode", ~test_split.baseline_episode_success),
    ):
        result[name] = ranking_metrics(test_split.returns[mask], scores[mask]) if np.any(mask) else None
        if result[name] is not None:
            result[name]["episodes"] = int(len(np.unique(test_split.episode_ids[mask])))
    return result


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
    normalization = compute_normalization(grouped["train"])
    pair_stats = pair_statistics(grouped)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "normalization.npz",
        observation_mean=normalization.observation_mean,
        observation_std=normalization.observation_std,
        action_mean=normalization.action_mean,
        action_std=normalization.action_std,
        observation_near_zero=normalization.observation_near_zero,
        action_near_zero=normalization.action_near_zero,
    )
    config = {
        "dataset": os.path.abspath(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "metadata_sha256": sha256_file(str(Path(args.dataset).with_suffix(".json"))),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed,
        "pair_margin_epsilon": args.pair_epsilon,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_states": args.batch_states,
        "max_epochs": args.max_epochs,
        "early_stopping_patience": args.patience,
        "architecture": {"hidden_dims": [128, 128], "activation": "gelu", "output_dim": 1},
        "normalization_source": "train split only",
        "training_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(output_dir / "config.json", "w") as file:
        json.dump(config, file, indent=2, sort_keys=True)

    print("dataset_validation=passed")
    print(f"dataset_shapes observation={arrays['observation'].shape} action={arrays['candidate_action_chunk'].shape}")
    print(f"pair_statistics={json.dumps(json_ready(pair_stats), sort_keys=True)}")
    print(
        f"normalization observation_mean/std={normalization.observation_mean.shape}/{normalization.observation_std.shape} "
        f"action_mean/std={normalization.action_mean.shape}/{normalization.action_std.shape} "
        f"near_zero_observation={normalization.observation_near_zero.tolist()} "
        f"near_zero_action={normalization.action_near_zero.tolist()}"
    )

    variant_results = {}
    variant_predictions = {}
    for variant in VARIANTS:
        result, predictions = train_variant(
            variant,
            grouped,
            normalization,
            output_dir,
            args.seed,
            args.learning_rate,
            args.weight_decay,
            args.pair_epsilon,
            args.batch_states,
            args.max_epochs,
            args.patience,
        )
        variant_results[variant] = result
        variant_predictions[variant] = predictions
        print(
            f"completed variant={variant} best_epoch={result['best_epoch']} "
            f"test_pairwise={result['evaluations']['test']['pairwise_accuracy']:.6f} "
            f"test_regret={result['evaluations']['test']['top1_regret']:.6f}",
            flush=True,
        )

    critic = {
        split: ranking_metrics(values.returns, values.critic_scores)
        for split, values in grouped.items()
    }
    test_subgroups = {
        "critic": subgroup_metrics(grouped["test"], grouped["test"].critic_scores),
        "selector": subgroup_metrics(grouped["test"], variant_predictions["full"]["test"]),
    }
    full_test = variant_results["full"]["evaluations"]["test"]
    full_validation = variant_results["full"]["evaluations"]["validation"]
    gates = {
        "test_pairwise_plus_0_05": full_test["pairwise_accuracy"] >= critic["test"]["pairwise_accuracy"] + 0.05,
        "test_pairwise_ge_0_65": full_test["pairwise_accuracy"] >= 0.65,
        "test_spearman_improved": full_test["spearman"] is not None and full_test["spearman"] > critic["test"]["spearman"],
        "test_regret_reduced_20_percent": full_test["top1_regret"] <= 0.8 * critic["test"]["top1_regret"],
        "test_non_tied_regret_reduced_20_percent": full_test["non_tied_top1_regret"] <= 0.8 * critic["test"]["non_tied_top1_regret"],
        "validation_pairwise_improved": full_validation["pairwise_accuracy"] > critic["validation"]["pairwise_accuracy"],
        "shuffle_near_random": abs(variant_results["shuffle"]["evaluations"]["test"]["pairwise_accuracy"] - 0.5) <= 0.10,
        "observation_only_near_random": abs(variant_results["observation_only"]["evaluations"]["test"]["pairwise_accuracy"] - 0.5) <= 0.01,
        "observation_only_score_spread_near_zero": variant_results["observation_only"]["evaluations"]["test"]["mean_score_spread"] <= 1e-6,
    }
    gates["all_passed"] = all(gates.values())
    metrics = {
        "config": config,
        "dataset_validation": {
            "pair_statistics": pair_stats,
            "normalization": json_ready(normalization),
        },
        "critic": critic,
        "variants": variant_results,
        "test_subgroups": test_subgroups,
        "real_environment_reranking_gates": gates,
    }
    with open(output_dir / "metrics.json", "w") as file:
        json.dump(json_ready(metrics), file, indent=2, sort_keys=True)
    print(f"critic_metrics={json.dumps(json_ready(critic), sort_keys=True)}")
    print(f"test_subgroups={json.dumps(json_ready(test_subgroups), sort_keys=True)}")
    print(f"reranking_gates={json.dumps(json_ready(gates), sort_keys=True)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
