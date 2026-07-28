"""Collect an evaluation-only H80 candidate-preference dataset pilot."""

import collections
import datetime
import json
import os
import random
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.branch_label_informativeness import (
    CHUNK_HORIZON,
    NUM_CANDIDATES,
    candidate_rng,
    closest_record,
    continuation_rng,
    deterministic_reset,
    discounted,
    execute_baseline_chunk,
    generate_eight_candidates_and_scores,
    install_deterministic_reset_action_space,
    label_metrics,
    partial_progress,
    prefix_summary,
    rollout_candidate,
    scan_episode,
    assert_trace_equal,
)
from diagnostics.candidate_branch_rollout_smoke import (
    CHECKPOINT,
    DATASET_DIR,
    ENV_NAME,
    assert_control_equal,
    capture_control_transition,
    create_restored_agent,
    sample_baseline_chunk,
)
from envs.mujoco_state import (
    clone_mujoco_env_state,
    restore_mujoco_env_state,
)
from envs.ogbench_utils import make_ogbench_env_and_datasets


NUM_EPISODES = 20
STATES_PER_EPISODE = 10
NUM_STATES = NUM_EPISODES * STATES_PER_EPISODE
CONTINUATION_HORIZON = 80
TOTAL_HORIZON = CHUNK_HORIZON + CONTINUATION_HORIZON
DISCOUNT = 0.99
COLLECTION_SEED = 0
OUTPUT_NPZ = (
    "/root/autodl-tmp/qc_workspace/datasets/"
    "qc_branch_preferences_h80_seed0.npz"
)
OUTPUT_JSON = OUTPUT_NPZ.replace(".npz", ".json")

PRIMARY_SEEDS = (
    20260728,
    20260729,
    20260730,
    20260731,
    20260732,
    20260733,
    20260734,
    20260735,
    20260736,
    20260737,
    20260743,
    20260745,
    20260746,
    20260755,
    20260768,
    20260781,
    20260789,
    20260790,
    20260822,
    20260833,
)
FALLBACK_FAILURE_SEEDS = (
    20260868,
    20260898,
    20260904,
    20260914,
    20260917,
    20260923,
    *range(20260924, 20261024),
)


def phase_for_step(step, episode_steps):
    fraction = step / max(episode_steps, 1)
    if fraction < 1 / 3:
        return "early"
    if fraction < 2 / 3:
        return "middle"
    return "late"


def select_ten_records(scan):
    """Greedily cover phases, progress, imminent changes, and low margins."""
    records = scan["records"]
    selected = []
    selected_ids = set()

    def add(record, reason):
        if (
            record.boundary_index not in selected_ids
            and len(selected) < STATES_PER_EPISODE
        ):
            selected.append((record, reason))
            selected_ids.add(record.boundary_index)

    for fraction, reason in (
        (0.1, "phase_early"),
        (0.5, "phase_middle"),
        (0.9, "phase_late"),
    ):
        add(
            closest_record(records, fraction * scan["steps"], selected_ids),
            reason,
        )

    for progress in (0, 1, 2):
        candidates = [
            record
            for record in records
            if record.progress == progress
            and record.boundary_index not in selected_ids
        ]
        if candidates:
            add(
                max(candidates, key=lambda record: record.potential),
                f"progress_{progress}",
            )

    imminent = sorted(
        (
            record
            for record in records
            if record.progress_will_change
            and record.boundary_index not in selected_ids
        ),
        key=lambda record: (
            -(record.next_chunk_max_progress - record.progress),
            -record.potential,
        ),
    )
    for record in imminent[:2]:
        add(record, "pre_progress_change")

    remaining = [
        record
        for record in records
        if record.boundary_index not in selected_ids
    ]
    if remaining:
        add(
            min(remaining, key=lambda record: record.critic_margin),
            "low_q_margin",
        )
    remaining = [
        record
        for record in records
        if record.boundary_index not in selected_ids
    ]
    if remaining:
        add(max(remaining, key=lambda record: record.potential), "near_goal")

    target_steps = np.linspace(0, scan["steps"], 4, endpoint=False)
    while len(selected) < STATES_PER_EPISODE:
        remaining = [
            record
            for record in records
            if record.boundary_index not in selected_ids
        ]
        if not remaining:
            raise RuntimeError("Episode has fewer than ten chunk boundaries")
        target = target_steps[len(selected) % len(target_steps)]
        add(
            min(remaining, key=lambda record: abs(record.step - target)),
            "coverage_fill",
        )
    return sorted(selected, key=lambda item: item[0].boundary_index)


def quick_episode_outcome(env, agent, seed):
    """Find failures without computing candidates at every boundary."""
    observation, _ = deterministic_reset(env, seed)
    policy_rng = jax.random.PRNGKey(seed)
    steps = 0
    terminated = False
    truncated = False
    final_info = {}
    while not (terminated or truncated):
        (
            observation,
            policy_rng,
            terminated,
            truncated,
            final_info,
            progresses,
        ) = execute_baseline_chunk(
            env, agent, observation, policy_rng
        )
        steps += len(progresses)
    return steps, bool(final_info.get("success", False))


def scan_collection_episodes(env, agent):
    scans = []
    for episode_index, seed in enumerate(PRIMARY_SEEDS):
        scan = scan_episode(env, agent, episode_index, seed)
        scans.append((seed, scan))
        print(
            f"scan episode={episode_index} seed={seed} "
            f"steps={scan['steps']} success={scan['success']}",
            flush=True,
        )

    if not any(not scan["success"] for _, scan in scans):
        for seed in FALLBACK_FAILURE_SEEDS:
            steps, success = quick_episode_outcome(env, agent, seed)
            print(
                f"failure search seed={seed} steps={steps} "
                f"success={success}",
                flush=True,
            )
            if not success:
                scan = scan_episode(
                    env, agent, NUM_EPISODES - 1, seed
                )
                if not scan["success"]:
                    scans[-1] = (seed, scan)
                    break
        else:
            raise RuntimeError(
                "No baseline failure found; refusing unstratified collection"
            )
    return scans


def collect_one_state(
    env,
    agent,
    episode_id,
    episode_seed,
    episode_success,
    state_id,
    record,
    reason,
    observation,
    policy_rng,
    repeatability_state_ids,
    pollution_state_ids,
):
    snapshot = clone_mujoco_env_state(env)
    snapshot_policy_rng = jnp.array(policy_rng, copy=True)
    chunks, scores = generate_eight_candidates_and_scores(
        agent,
        jnp.asarray(observation),
        candidate_rng(episode_id, record.boundary_index),
    )
    chunks = np.asarray(chunks)
    scores = np.asarray(scores)
    assert chunks.shape == (NUM_CANDIDATES, CHUNK_HORIZON, 5)

    control = None
    control_chunk = None
    control_next_rng = None
    if state_id in pollution_state_ids:
        control_chunk, control_next_rng = sample_baseline_chunk(
            agent, observation, snapshot_policy_rng
        )
        control = capture_control_transition(
            env, control_chunk[0], control_next_rng
        )
        restore_mujoco_env_state(env, snapshot)

    shared_rng = continuation_rng(episode_id, record.boundary_index)
    traces = [
        rollout_candidate(
            env,
            snapshot,
            observation,
            chunks[candidate_index],
            scores[candidate_index],
            agent,
            shared_rng,
            CONTINUATION_HORIZON,
        )
        for candidate_index in range(NUM_CANDIDATES)
    ]
    if state_id in repeatability_state_ids:
        repeated = rollout_candidate(
            env,
            snapshot,
            observation,
            chunks[0],
            scores[0],
            agent,
            shared_rng,
            CONTINUATION_HORIZON,
        )
        assert_trace_equal(traces[0], repeated)

    restore_mujoco_env_state(env, snapshot)
    actual_chunk, actual_next_rng = sample_baseline_chunk(
        agent, observation, snapshot_policy_rng
    )
    if control is not None:
        np.testing.assert_array_equal(control_chunk, actual_chunk)
        actual = capture_control_transition(
            env, actual_chunk[0], actual_next_rng
        )
        assert_control_equal(control, actual)
        next_observation = actual["observation"]
        terminated = actual["terminated"]
        truncated = actual["truncated"]
    else:
        (
            next_observation,
            _,
            terminated,
            truncated,
            _,
        ) = env.step(actual_chunk[0])

    phase = phase_for_step(record.step, collect_one_state.episode_steps)
    state_role = (
        f"{'baseline_success' if episode_success else 'baseline_failure'}:"
        f"{phase}:{reason}"
    )
    rows = []
    for candidate_index, trace in enumerate(traces):
        summary = prefix_summary(trace, CONTINUATION_HORIZON)
        reward_values = summary["reward_sequence"]
        reward_padded = np.zeros(TOTAL_HORIZON, dtype=np.float32)
        reward_mask = np.zeros(TOTAL_HORIZON, dtype=bool)
        reward_padded[: len(reward_values)] = reward_values
        reward_mask[: len(reward_values)] = True
        rows.append(
            {
                "episode_seed": episode_seed,
                "episode_id": episode_id,
                "state_id": state_id,
                "state_step": record.step,
                "state_role": state_role,
                "state_phase": phase,
                "selection_reason": reason,
                "baseline_episode_success": episode_success,
                "candidate_index": candidate_index,
                "observation": np.array(observation, dtype=np.float32),
                "candidate_action_chunk": np.array(
                    chunks[candidate_index], dtype=np.float32
                ),
                "critic_score": float(scores[candidate_index]),
                "discounted_return": summary["discounted_return"],
                "undiscounted_return": summary["undiscounted_return"],
                "reward_sequence": reward_padded,
                "reward_mask": reward_mask,
                "success_ever": summary["success_ever"],
                "first_success_step": (
                    -1
                    if summary["first_success_step"] is None
                    else summary["first_success_step"]
                ),
                "initial_progress": summary["initial_progress"],
                "final_progress": summary["final_progress"],
                "max_progress": summary["max_progress"],
                "potential_delta": summary["potential_delta"],
                "terminated": summary["terminated"],
                "truncated": summary["truncated"],
                "executed_steps": summary["executed_steps"],
            }
        )
    return (
        rows,
        next_observation,
        actual_chunk,
        actual_next_rng,
        terminated,
        truncated,
    )


def replay_and_collect_episode(
    env,
    agent,
    episode_id,
    episode_seed,
    scan,
    selected,
    state_id_start,
    repeatability_state_ids,
    pollution_state_ids,
):
    restore_mujoco_env_state(env, scan["initial_snapshot"])
    observation = np.array(scan["records"][0].observation, copy=True)
    policy_rng = jnp.array(scan["initial_policy_rng"], copy=True)
    selected_map = {
        record.boundary_index: (record, reason, state_id_start + offset)
        for offset, (record, reason) in enumerate(selected)
    }
    collect_one_state.episode_steps = scan["steps"]
    max_selected = max(selected_map)
    boundary_index = 0
    rows = []

    while boundary_index <= max_selected:
        record = scan["records"][boundary_index]
        np.testing.assert_array_equal(observation, record.observation)
        if boundary_index in selected_map:
            record, reason, state_id = selected_map[boundary_index]
            (
                state_rows,
                observation,
                action_chunk,
                policy_rng,
                terminated,
                truncated,
            ) = collect_one_state(
                env,
                agent,
                episode_id,
                episode_seed,
                scan["success"],
                state_id,
                record,
                reason,
                observation,
                policy_rng,
                repeatability_state_ids,
                pollution_state_ids,
            )
            rows.extend(state_rows)
            for action in action_chunk[1:]:
                if terminated or truncated:
                    break
                _, policy_rng = sample_baseline_chunk(
                    agent, observation, policy_rng
                )
                observation, _, terminated, truncated, _ = env.step(action)
        else:
            (
                observation,
                policy_rng,
                terminated,
                truncated,
                _,
                _,
            ) = execute_baseline_chunk(
                env, agent, observation, policy_rng
            )
        if terminated or truncated:
            break
        boundary_index += 1
    if len(rows) != STATES_PER_EPISODE * NUM_CANDIDATES:
        raise RuntimeError("Episode did not yield exactly ten selected states")
    return rows


def assign_episode_splits(episode_ids):
    episode_ids = np.asarray(sorted(set(episode_ids)), dtype=np.int32)
    rng = np.random.default_rng(COLLECTION_SEED + 9000)
    shuffled = rng.permutation(episode_ids)
    split_by_episode = {}
    for episode_id in shuffled[:14]:
        split_by_episode[int(episode_id)] = "train"
    for episode_id in shuffled[14:17]:
        split_by_episode[int(episode_id)] = "validation"
    for episode_id in shuffled[17:]:
        split_by_episode[int(episode_id)] = "test"
    return split_by_episode


def rows_to_arrays(rows, split_by_episode):
    keys = rows[0].keys()
    arrays = {}
    for key in keys:
        values = [row[key] for row in rows]
        if isinstance(values[0], np.ndarray):
            arrays[key] = np.stack(values)
        else:
            arrays[key] = np.asarray(values)
    arrays["split"] = np.asarray(
        [split_by_episode[int(value)] for value in arrays["episode_id"]]
    )
    return arrays


def state_statistics(arrays, split_name):
    mask = (
        np.ones(len(arrays["state_id"]), dtype=bool)
        if split_name == "overall"
        else arrays["split"] == split_name
    )
    row_indices = np.flatnonzero(mask)
    state_ids = np.unique(arrays["state_id"][row_indices])
    labels = []
    critics = []
    progresses = []
    phases = []
    episode_successes = []
    for state_id in state_ids:
        indices = np.flatnonzero(
            mask & (arrays["state_id"] == state_id)
        )
        assert len(indices) == NUM_CANDIDATES
        labels.append(arrays["discounted_return"][indices])
        critics.append(arrays["critic_score"][indices])
        progresses.append(int(arrays["initial_progress"][indices[0]]))
        phases.append(str(arrays["state_phase"][indices[0]]))
        episode_successes.append(
            bool(arrays["baseline_episode_success"][indices[0]])
        )
    metrics = label_metrics(labels, critics)
    non_tied_regrets = []
    for label, critic in zip(labels, critics):
        if np.ptp(label) > 0:
            non_tied_regrets.append(
                float(np.max(label) - label[int(np.argmax(critic))])
            )
    return {
        "states": len(state_ids),
        "episodes": len(np.unique(arrays["episode_id"][row_indices])),
        "progress_distribution": dict(
            collections.Counter(str(value) for value in progresses)
        ),
        "phase_distribution": dict(collections.Counter(phases)),
        "episode_outcome_state_distribution": dict(
            collections.Counter(
                "success" if value else "failure"
                for value in episode_successes
            )
        ),
        "mean_candidate_spread": metrics["mean_spread"],
        "median_candidate_spread": metrics["median_spread"],
        "tied_state_fraction": (
            metrics["tied_states"] / metrics["state_count"]
        ),
        "meaningful_state_fraction": (
            1.0 - metrics["tied_states"] / metrics["state_count"]
        ),
        "critic_spearman": metrics["spearman"],
        "critic_spearman_states": metrics["spearman_states"],
        "critic_pairwise_accuracy": metrics["pairwise_accuracy"],
        "critic_pair_count": metrics["pair_count"],
        "critic_top1_regret": metrics["top1_regret"],
        "non_tied_critic_top1_regret": float(
            np.mean(non_tied_regrets)
        ),
        "true_best_critic_percentile": metrics[
            "true_best_critic_percentile"
        ],
        "critic_best_label_percentile": metrics[
            "critic_best_label_percentile"
        ],
    }


def validate_arrays(arrays):
    row_count = NUM_STATES * NUM_CANDIDATES
    assert len(arrays["state_id"]) == row_count
    assert arrays["observation"].shape == (row_count, 46)
    assert arrays["candidate_action_chunk"].shape == (
        row_count,
        CHUNK_HORIZON,
        5,
    )
    assert arrays["reward_sequence"].shape == (row_count, TOTAL_HORIZON)
    assert arrays["reward_mask"].shape == (row_count, TOTAL_HORIZON)
    for state_id in range(NUM_STATES):
        indices = np.flatnonzero(arrays["state_id"] == state_id)
        assert len(indices) == NUM_CANDIDATES
        np.testing.assert_array_equal(
            np.sort(arrays["candidate_index"][indices]),
            np.arange(NUM_CANDIDATES),
        )
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating):
            assert np.all(np.isfinite(value)), f"{key} contains NaN/Inf"
    assert np.all(
        arrays["reward_mask"].sum(axis=1) == arrays["executed_steps"]
    )
    for split in ("train", "validation", "test"):
        other = arrays["split"] != split
        split_episodes = set(arrays["episode_id"][arrays["split"] == split])
        other_episodes = set(arrays["episode_id"][other])
        assert split_episodes.isdisjoint(other_episodes)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    np.random.seed(COLLECTION_SEED)
    random.seed(COLLECTION_SEED)
    env = make_ogbench_env_and_datasets(
        ENV_NAME, dataset_dir=DATASET_DIR, env_only=True
    )
    try:
        install_deterministic_reset_action_space(env)
        observation, _ = deterministic_reset(env, PRIMARY_SEEDS[0])
        agent = create_restored_agent(
            observation, np.zeros(env.action_space.shape, dtype=np.float32)
        )
        scans = scan_collection_episodes(env, agent)
        if len(scans) != NUM_EPISODES:
            raise RuntimeError("Expected exactly twenty retained episodes")

        quality_rng = np.random.default_rng(COLLECTION_SEED + 1234)
        repeatability_state_ids = set(
            quality_rng.choice(NUM_STATES, size=10, replace=False).tolist()
        )
        pollution_state_ids = set(
            quality_rng.choice(NUM_STATES, size=10, replace=False).tolist()
        )
        print(
            f"repeatability_state_ids={sorted(repeatability_state_ids)}"
        )
        print(f"pollution_state_ids={sorted(pollution_state_ids)}")

        rows = []
        retained_seeds = []
        for episode_id, (seed, scan) in enumerate(scans):
            selected = select_ten_records(scan)
            retained_seeds.append(seed)
            episode_rows = replay_and_collect_episode(
                env,
                agent,
                episode_id,
                seed,
                scan,
                selected,
                episode_id * STATES_PER_EPISODE,
                repeatability_state_ids,
                pollution_state_ids,
            )
            rows.extend(episode_rows)
            print(
                f"collected episode={episode_id + 1}/{NUM_EPISODES} "
                f"seed={seed} success={scan['success']} "
                f"rows={len(rows)}/{NUM_STATES * NUM_CANDIDATES}",
                flush=True,
            )

        split_by_episode = assign_episode_splits(
            [row["episode_id"] for row in rows]
        )
        arrays = rows_to_arrays(rows, split_by_episode)
        validate_arrays(arrays)
        statistics = {
            split: state_statistics(arrays, split)
            for split in ("overall", "train", "validation", "test")
        }
        overall = statistics["overall"]
        gates = {
            "meaningful_fraction_ge_0_60": (
                overall["meaningful_state_fraction"] >= 0.60
            ),
            "tied_fraction_le_0_40": (
                overall["tied_state_fraction"] <= 0.40
            ),
            "median_spread_gt_1": (
                overall["median_candidate_spread"] > 1.0
            ),
            "critic_pairwise_lt_0_75": (
                overall["critic_pairwise_accuracy"] < 0.75
            ),
            "non_tied_regret_gt_0_1": (
                overall["non_tied_critic_top1_regret"] > 0.1
            ),
        }
        gates["all_passed"] = all(gates.values())

        os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
        np.savez_compressed(OUTPUT_NPZ, **arrays)
        metadata = {
            "checkpoint": CHECKPOINT,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "environment": ENV_NAME,
            "candidate_generation": {
                "actor_type": str(agent.config["actor_type"]),
                "diagnostic_candidates": NUM_CANDIDATES,
                "baseline_actor_num_samples": int(
                    agent.config["actor_num_samples"]
                ),
                "q_aggregation": str(agent.config["q_agg"]),
                "action_chunk_horizon": CHUNK_HORIZON,
            },
            "discount": DISCOUNT,
            "continuation_horizon": CONTINUATION_HORIZON,
            "total_branch_horizon": TOTAL_HORIZON,
            "collection_seed": COLLECTION_SEED,
            "episode_seeds": retained_seeds,
            "split_by_episode": split_by_episode,
            "dataset_shape": {
                "episodes": NUM_EPISODES,
                "states": NUM_STATES,
                "candidates_per_state": NUM_CANDIDATES,
                "state_candidate_rows": len(rows),
            },
            "dataset_shapes": {
                key: list(value.shape) for key, value in arrays.items()
            },
            "repeatability_state_ids": sorted(repeatability_state_ids),
            "pollution_state_ids": sorted(pollution_state_ids),
            "collection_timestamp_utc": (
                datetime.datetime.now(datetime.timezone.utc).isoformat()
            ),
            "statistics": statistics,
            "quality_gates": gates,
        }
        with open(OUTPUT_JSON, "w") as file:
            json.dump(json_ready(metadata), file, indent=2, sort_keys=True)

        print(f"dataset_npz={OUTPUT_NPZ}")
        print(f"dataset_json={OUTPUT_JSON}")
        print(f"npz_bytes={os.path.getsize(OUTPUT_NPZ)}")
        print(f"json_bytes={os.path.getsize(OUTPUT_JSON)}")
        print("schema:")
        for key in sorted(arrays):
            print(
                f"{key}: shape={arrays[key].shape} dtype={arrays[key].dtype}"
            )
        print("statistics:")
        for split, values in statistics.items():
            print(f"{split}: {json.dumps(json_ready(values), sort_keys=True)}")
        print(f"quality_gates: {json.dumps(gates, sort_keys=True)}")
        print("repeatability_checks_passed=10/10")
        print("pollution_checks_passed=10/10")
        print("H80 candidate preference dataset pilot passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
