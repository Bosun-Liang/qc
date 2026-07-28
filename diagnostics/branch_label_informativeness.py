"""Evaluation-only informativeness study for true candidate branch labels.

The diagnostic uses exact MuJoCo snapshots and the unchanged QC best-of-N
policy.  It trains nothing and does not alter main.py or online collection.
"""

import dataclasses
import random

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.candidate_branch_rollout_smoke import (
    CHUNK_HORIZON,
    DATASET_DIR,
    ENV_NAME,
    assert_control_equal,
    capture_control_transition,
    create_restored_agent,
    cube_potential,
    sample_baseline_chunk,
)
from envs.mujoco_state import (
    clone_mujoco_env_state,
    restore_mujoco_env_state,
)
from envs.ogbench_utils import make_ogbench_env_and_datasets


NUM_CANDIDATES = 8
CONTINUATION_HORIZONS = (20, 40, 80)
MAX_CONTINUATION = max(CONTINUATION_HORIZONS)
DISCOUNT = 0.99
DIAGNOSTIC_SEED = 8128
MEANINGFUL_POTENTIAL_DELTA = 0.01

# These seeds come from the already completed 200-episode baseline evaluation.
# Prior labels are only sampling hints: actual success/failure is determined
# again in this process because reset internals/GPU execution can drift across
# independent runs.
EPISODES = (
    (20260729, "success_short"),
    (20260728, "success_medium"),
    (20260730, "success_long"),
    (20260743, "failure_partial"),
    (20260746, "failure_no_progress"),
)
FAILURE_SEARCH_SEEDS = (
    20260746,
    20260755,
    20260790,
    20260848,
    20260863,
    20260833,
    20260768,
    20260781,
    20260789,
    20260822,
)


def install_deterministic_reset_action_space(env):
    """Make OGBench internal reset-time action samples seedable.

    CubeEnv.action_space constructs a fresh Box on every access. Its two
    reset-time goal-stabilization samples otherwise use an untracked temporary
    Generator. This diagnostic-only override returns one persistent Box.
    """
    core = env.unwrapped
    fixed_space = core.action_space
    core._diagnostic_reset_action_space = fixed_space
    type(core).action_space = property(
        lambda instance: instance._diagnostic_reset_action_space
    )


def deterministic_reset(env, seed):
    env.unwrapped._diagnostic_reset_action_space.seed(seed)
    return env.reset(seed=seed)


@jax.jit
def generate_eight_candidates_and_scores(agent, observation, rng):
    """Exact existing actor-flow and critic path with eight diagnostics."""
    flat_action_dim = agent.config["action_dim"] * CHUNK_HORIZON
    noises = jax.random.normal(rng, (NUM_CANDIDATES, flat_action_dim))
    observations = jnp.repeat(
        observation[None, :], NUM_CANDIDATES, axis=0
    )
    actions = jnp.clip(
        agent.compute_flow_actions(observations, noises), -1, 1
    )
    qs = agent.network.select("critic")(observations, actions=actions)
    scores = (
        qs.min(axis=0)
        if agent.config["q_agg"] == "min"
        else qs.mean(axis=0)
    )
    return actions.reshape(NUM_CANDIDATES, CHUNK_HORIZON, 5), scores


@dataclasses.dataclass
class BoundaryRecord:
    boundary_index: int
    step: int
    observation: np.ndarray
    progress: int
    potential: float
    closest_distance: float
    critic_margin: float
    progress_will_change: bool = False
    next_chunk_max_progress: int = 0


@dataclasses.dataclass
class BranchTrace:
    rewards: np.ndarray
    actions: np.ndarray
    progresses: np.ndarray
    potentials: np.ndarray
    successes: np.ndarray
    terminated: bool
    truncated: bool
    initial_observation: np.ndarray
    final_observation: np.ndarray
    initial_qpos: np.ndarray
    final_qpos: np.ndarray
    initial_qvel: np.ndarray
    final_qvel: np.ndarray
    critic_score: float
    candidate_chunk: np.ndarray


def partial_progress(env):
    """Number of cubes currently within the environment's 4-cm threshold."""
    return int(sum(env.unwrapped._compute_successes()))


def cube_distances(env):
    core = env.unwrapped
    distances = []
    for index in range(core._num_cubes):
        xyz = core.data.joint(f"object_joint_{index}").qpos[:3]
        target = core.data.mocap_pos[core._cube_target_mocap_ids[index]]
        distances.append(float(np.linalg.norm(xyz - target)))
    return np.asarray(distances)


def candidate_rng(episode_index, boundary_index):
    root = jax.random.PRNGKey(DIAGNOSTIC_SEED + episode_index)
    return jax.random.fold_in(root, boundary_index)


def continuation_rng(episode_index, boundary_index):
    root = jax.random.PRNGKey(DIAGNOSTIC_SEED + 10_000 + episode_index)
    return jax.random.fold_in(root, boundary_index)


def critic_margin(scores):
    sorted_scores = np.sort(np.asarray(scores))
    return float(sorted_scores[-1] - sorted_scores[-2])


def execute_baseline_chunk(env, agent, observation, policy_rng):
    """Execute one baseline chunk with evaluation.py's exact RNG schedule."""
    action_chunk, policy_rng = sample_baseline_chunk(
        agent, observation, policy_rng
    )
    progresses = []
    terminated = False
    truncated = False
    info = {}
    for action_index, action in enumerate(action_chunk):
        if action_index > 0:
            # evaluation.py calls sample_actions even while its queue is
            # non-empty, discarding the returned chunk but advancing the key.
            _, policy_rng = sample_baseline_chunk(
                agent, observation, policy_rng
            )
        observation, _, terminated, truncated, info = env.step(action)
        progresses.append(partial_progress(env))
        if terminated or truncated:
            break
    return (
        observation,
        policy_rng,
        terminated,
        truncated,
        info,
        progresses,
    )


def scan_episode(env, agent, episode_index, seed):
    """Run one baseline episode and retain compact chunk-boundary metadata."""
    observation, _ = deterministic_reset(env, seed)
    initial_snapshot = clone_mujoco_env_state(env)
    initial_policy_rng = jax.random.PRNGKey(seed)
    policy_rng = jnp.array(initial_policy_rng, copy=True)
    records = []
    step = 0
    terminated = False
    truncated = False
    final_info = {}

    while not (terminated or truncated):
        progress = partial_progress(env)
        _, scores = generate_eight_candidates_and_scores(
            agent,
            jnp.asarray(observation),
            candidate_rng(episode_index, len(records)),
        )
        distances = cube_distances(env)
        record = BoundaryRecord(
            boundary_index=len(records),
            step=step,
            observation=np.array(observation, copy=True),
            progress=progress,
            potential=cube_potential(env),
            closest_distance=float(np.min(distances)),
            critic_margin=critic_margin(scores),
            next_chunk_max_progress=progress,
        )
        records.append(record)

        (
            observation,
            policy_rng,
            terminated,
            truncated,
            final_info,
            chunk_progresses,
        ) = execute_baseline_chunk(
            env, agent, observation, policy_rng
        )
        step += len(chunk_progresses)
        record.next_chunk_max_progress = max(
            [record.progress, *chunk_progresses]
        )
        record.progress_will_change = any(
            value != record.progress for value in chunk_progresses
        )

    return {
        "records": records,
        "steps": step,
        "success": bool(final_info.get("success", False)),
        "initial_snapshot": initial_snapshot,
        "initial_policy_rng": initial_policy_rng,
    }


def closest_record(records, target_step, excluded):
    candidates = [
        record for record in records if record.boundary_index not in excluded
    ]
    return min(candidates, key=lambda record: abs(record.step - target_step))


def select_records(scan):
    """Select early/middle/late plus one control-relevant special state."""
    records = scan["records"]
    episode_steps = scan["steps"]
    selected = []
    excluded = set()
    for fraction in (0.1, 0.5, 0.9):
        record = closest_record(records, fraction * episode_steps, excluded)
        selected.append((record, ("early", "middle", "late")[len(selected)]))
        excluded.add(record.boundary_index)

    remaining = [
        record for record in records if record.boundary_index not in excluded
    ]
    pre_progress = [
        record for record in remaining if record.progress_will_change
    ]
    partial = [record for record in remaining if record.progress > 0]
    if pre_progress:
        special = max(
            pre_progress,
            key=lambda record: (
                record.next_chunk_max_progress - record.progress,
                record.potential,
            ),
        )
        reason = "pre_progress_change"
    elif partial:
        special = max(
            partial, key=lambda record: (record.progress, record.potential)
        )
        reason = "partial_progress"
    else:
        near = max(remaining, key=lambda record: record.potential)
        low_margin = min(
            remaining, key=lambda record: record.critic_margin
        )
        if near.potential > records[0].potential + 0.05:
            special = near
            reason = "near_goal"
        else:
            special = low_margin
            reason = "low_q_margin"
    selected.append((special, reason))
    return sorted(selected, key=lambda item: item[0].boundary_index)


def rollout_candidate(
    env,
    snapshot,
    observation,
    candidate_chunk,
    score,
    agent,
    initial_continuation_rng,
    continuation_limit,
    until_terminal=False,
):
    """Roll out one candidate and an identical-key baseline continuation."""
    restore_mujoco_env_state(env, snapshot)
    current_observation = np.array(observation, copy=True)
    initial_qpos = env.unwrapped.data.qpos.copy()
    initial_qvel = env.unwrapped.data.qvel.copy()
    rewards = []
    actions = []
    progresses = [partial_progress(env)]
    potentials = [cube_potential(env)]
    successes = []
    terminated = False
    truncated = False

    def take_step(action):
        nonlocal current_observation, terminated, truncated
        current_observation, reward, terminated, truncated, info = env.step(
            np.clip(action, -1, 1)
        )
        progress = partial_progress(env)
        # CubeEnv.compute_reward is exactly progress - num_cubes.
        assert float(reward) == float(progress - env.unwrapped._num_cubes)
        rewards.append(float(reward))
        actions.append(np.array(action, copy=True))
        progresses.append(progress)
        potentials.append(cube_potential(env))
        successes.append(bool(info.get("success", False)))

    for action in candidate_chunk:
        take_step(action)
        if terminated or truncated:
            break

    local_rng = jnp.array(initial_continuation_rng, copy=True)
    action_queue = []
    continuation_steps = 0
    while not (terminated or truncated):
        if not until_terminal and continuation_steps >= continuation_limit:
            break
        if until_terminal and continuation_steps >= 1000:
            raise RuntimeError("Terminal rollout exceeded safety limit")
        sampled_chunk, local_rng = sample_baseline_chunk(
            agent, current_observation, local_rng
        )
        if not action_queue:
            action_queue.extend(sampled_chunk)
        action = action_queue.pop(0)
        take_step(action)
        continuation_steps += 1

    return BranchTrace(
        rewards=np.asarray(rewards, dtype=np.float64),
        actions=np.asarray(actions, dtype=np.float32),
        progresses=np.asarray(progresses, dtype=np.int32),
        potentials=np.asarray(potentials, dtype=np.float64),
        successes=np.asarray(successes, dtype=bool),
        terminated=bool(terminated),
        truncated=bool(truncated),
        initial_observation=np.array(observation, copy=True),
        final_observation=np.array(current_observation, copy=True),
        initial_qpos=initial_qpos,
        final_qpos=env.unwrapped.data.qpos.copy(),
        initial_qvel=initial_qvel,
        final_qvel=env.unwrapped.data.qvel.copy(),
        critic_score=float(score),
        candidate_chunk=np.array(candidate_chunk, copy=True),
    )


def assert_trace_equal(first, second):
    for name in (
        "rewards",
        "actions",
        "progresses",
        "potentials",
        "successes",
        "final_observation",
        "final_qpos",
        "final_qvel",
        "candidate_chunk",
    ):
        np.testing.assert_array_equal(
            getattr(first, name), getattr(second, name)
        )
    assert first.terminated == second.terminated
    assert first.truncated == second.truncated
    assert first.critic_score == second.critic_score


def discounted(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(DISCOUNT ** np.arange(len(values)) * values))


def prefix_summary(trace, continuation_horizon):
    total_horizon = CHUNK_HORIZON + continuation_horizon
    count = min(total_horizon, len(trace.rewards))
    rewards = trace.rewards[:count]
    progresses = trace.progresses[: count + 1]
    potentials = trace.potentials[: count + 1]
    successes = trace.successes[:count]
    success_indices = np.flatnonzero(successes)
    return {
        "reward_sequence": rewards,
        "unique_rewards": np.unique(rewards),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "reward_std": float(np.std(rewards)),
        "discounted_return": discounted(rewards),
        "undiscounted_return": float(np.sum(rewards)),
        "chunk_return": discounted(rewards[:CHUNK_HORIZON]),
        "continuation_return": discounted(rewards[CHUNK_HORIZON:]),
        "success_final": bool(successes[-1]) if len(successes) else False,
        "success_ever": bool(np.any(successes)),
        "first_success_step": (
            int(success_indices[0] + 1) if len(success_indices) else None
        ),
        "terminated": trace.terminated and count == len(trace.rewards),
        "truncated": trace.truncated and count == len(trace.rewards),
        "executed_steps": count,
        "initial_progress": int(progresses[0]),
        "final_progress": int(progresses[-1]),
        "max_progress": int(np.max(progresses)),
        "progress_transition_count": int(
            np.count_nonzero(np.diff(progresses))
        ),
        "initial_potential": float(potentials[0]),
        "final_potential": float(potentials[-1]),
        "potential_delta": float(potentials[-1] - potentials[0]),
        "critic_score": trace.critic_score,
        "candidate_chunk": trace.candidate_chunk,
    }


def terminal_summary(trace):
    successes = trace.successes
    success_indices = np.flatnonzero(successes)
    return {
        "success": bool(np.any(successes)),
        "first_success_step": (
            int(success_indices[0] + 1) if len(success_indices) else None
        ),
        "discounted_return": discounted(trace.rewards),
        "undiscounted_return": float(np.sum(trace.rewards)),
        "executed_steps": len(trace.rewards),
        "terminated": trace.terminated,
        "truncated": trace.truncated,
    }


def percentile(value, values):
    values = np.asarray(values)
    less = np.sum(values < value)
    equal = np.sum(values == value)
    if len(values) == 1:
        return 1.0
    return float((less + 0.5 * (equal - 1)) / (len(values) - 1))


def rank_correlation(left, right):
    def ranks(values):
        values = np.asarray(values)
        result = np.empty(len(values), dtype=np.float64)
        order = np.argsort(values, kind="stable")
        start = 0
        while start < len(values):
            end = start + 1
            while (
                end < len(values)
                and values[order[end]] == values[order[start]]
            ):
                end += 1
            result[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return result

    left = ranks(left)
    right = ranks(right)
    if np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def label_metrics(state_values, state_critics, meaningful_threshold=0.0):
    spreads = []
    correlations = []
    regrets = []
    true_best_critic_percentiles = []
    critic_best_label_percentiles = []
    tied = 0
    meaningful = 0
    pair_correct = 0.0
    pair_count = 0

    for labels, critics in zip(state_values, state_critics):
        labels = np.asarray(labels, dtype=np.float64)
        critics = np.asarray(critics, dtype=np.float64)
        spread = float(np.max(labels) - np.min(labels))
        spreads.append(spread)
        tied += spread == 0
        meaningful += spread > meaningful_threshold
        correlation = rank_correlation(critics, labels)
        if np.isfinite(correlation):
            correlations.append(correlation)

        critic_best = int(np.argmax(critics))
        true_best_mask = labels == np.max(labels)
        regrets.append(float(np.max(labels) - labels[critic_best]))
        true_best_critic_percentiles.append(
            max(
                percentile(critics[index], critics)
                for index in np.flatnonzero(true_best_mask)
            )
        )
        critic_best_label_percentiles.append(
            percentile(labels[critic_best], labels)
        )

        for left in range(len(labels)):
            for right in range(left + 1, len(labels)):
                label_delta = labels[left] - labels[right]
                if label_delta == 0:
                    continue
                critic_delta = critics[left] - critics[right]
                if critic_delta == 0:
                    pair_correct += 0.5
                elif np.sign(label_delta) == np.sign(critic_delta):
                    pair_correct += 1.0
                pair_count += 1

    return {
        "mean_spread": float(np.mean(spreads)),
        "median_spread": float(np.median(spreads)),
        "tied_states": tied,
        "state_count": len(spreads),
        "meaningful_fraction": float(meaningful / len(spreads)),
        "spearman": (
            float(np.mean(correlations)) if correlations else np.nan
        ),
        "spearman_states": len(correlations),
        "pairwise_accuracy": (
            float(pair_correct / pair_count) if pair_count else np.nan
        ),
        "pair_count": pair_count,
        "top1_regret": float(np.mean(regrets)),
        "true_best_critic_percentile": float(
            np.mean(true_best_critic_percentiles)
        ),
        "critic_best_label_percentile": float(
            np.mean(critic_best_label_percentiles)
        ),
    }


def analyze_selected_state(
    env,
    agent,
    episode_index,
    episode_seed,
    episode_role,
    record,
    reason,
    observation,
    policy_rng,
    terminal_extra,
):
    snapshot = clone_mujoco_env_state(env)
    snapshot_policy_rng = jnp.array(policy_rng, copy=True)
    chunks, scores = generate_eight_candidates_and_scores(
        agent,
        jnp.asarray(observation),
        candidate_rng(episode_index, record.boundary_index),
    )
    chunks = np.asarray(chunks)
    scores = np.asarray(scores)
    assert chunks.shape == (NUM_CANDIDATES, CHUNK_HORIZON, 5)

    baseline_chunk, control_next_rng = sample_baseline_chunk(
        agent, observation, snapshot_policy_rng
    )
    control = capture_control_transition(
        env, baseline_chunk[0], control_next_rng
    )
    restore_mujoco_env_state(env, snapshot)

    shared_continuation_rng = continuation_rng(
        episode_index, record.boundary_index
    )
    traces = [
        rollout_candidate(
            env,
            snapshot,
            observation,
            chunks[index],
            scores[index],
            agent,
            shared_continuation_rng,
            MAX_CONTINUATION,
        )
        for index in range(NUM_CANDIDATES)
    ]
    repeated = rollout_candidate(
        env,
        snapshot,
        observation,
        chunks[0],
        scores[0],
        agent,
        shared_continuation_rng,
        MAX_CONTINUATION,
    )
    assert_trace_equal(traces[0], repeated)

    terminal_traces = None
    if terminal_extra:
        terminal_traces = [
            rollout_candidate(
                env,
                snapshot,
                observation,
                chunks[index],
                scores[index],
                agent,
                shared_continuation_rng,
                continuation_limit=0,
                until_terminal=True,
            )
            for index in range(NUM_CANDIDATES)
        ]

    restore_mujoco_env_state(env, snapshot)
    actual_chunk, actual_next_rng = sample_baseline_chunk(
        agent, observation, snapshot_policy_rng
    )
    np.testing.assert_array_equal(baseline_chunk, actual_chunk)
    actual = capture_control_transition(
        env, actual_chunk[0], actual_next_rng
    )
    assert_control_equal(control, actual)

    summaries = {
        horizon: [
            prefix_summary(trace, horizon) for trace in traces
        ]
        for horizon in CONTINUATION_HORIZONS
    }
    terminal_summaries = (
        [terminal_summary(trace) for trace in terminal_traces]
        if terminal_traces is not None
        else None
    )
    return {
        "episode_seed": episode_seed,
        "episode_role": episode_role,
        "step": record.step,
        "phase_reason": reason,
        "initial_progress": record.progress,
        "initial_potential": record.potential,
        "closest_distance": record.closest_distance,
        "critic_margin": critic_margin(scores),
        "critic_scores": scores,
        "summaries": summaries,
        "terminal": terminal_summaries,
        "pollution_passed": True,
        "repeat_passed": True,
    }, actual, actual_chunk, actual_next_rng


def replay_and_analyze(
    env,
    agent,
    episode_index,
    seed,
    role,
    scan,
    selected,
):
    restore_mujoco_env_state(env, scan["initial_snapshot"])
    observation = np.array(scan["records"][0].observation, copy=True)
    policy_rng = jnp.array(scan["initial_policy_rng"], copy=True)
    selected_by_index = {
        record.boundary_index: (record, reason)
        for record, reason in selected
    }
    terminal_boundary = min(
        selected,
        key=lambda item: abs(
            item[0].step - 0.5 * scan["steps"]
        ),
    )[0].boundary_index
    results = []
    max_selected = max(selected_by_index)
    boundary_index = 0

    while boundary_index <= max_selected:
        record = scan["records"][boundary_index]
        np.testing.assert_array_equal(observation, record.observation)
        if boundary_index in selected_by_index:
            _, reason = selected_by_index[boundary_index]
            terminal_extra = (
                (episode_index == 0 and boundary_index == terminal_boundary)
                or (
                    role == "baseline_failure"
                    and reason == "late"
                )
            )
            result, actual, action_chunk, policy_rng = (
                analyze_selected_state(
                    env,
                    agent,
                    episode_index,
                    seed,
                    role,
                    record,
                    reason,
                    observation,
                    policy_rng,
                    terminal_extra,
                )
            )
            results.append(result)
            observation = actual["observation"]
            terminated = actual["terminated"]
            truncated = actual["truncated"]
            for action_index, action in enumerate(action_chunk[1:], start=1):
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
    return results


def collect_label_table(states):
    critics = [state["critic_scores"] for state in states]
    label_sets = {}
    thresholds = {}
    for horizon in CONTINUATION_HORIZONS:
        prefix = f"cont{horizon}_total{CHUNK_HORIZON + horizon}"
        label_sets[f"{prefix}_discounted_return"] = [
            [item["discounted_return"] for item in state["summaries"][horizon]]
            for state in states
        ]
        label_sets[f"{prefix}_undiscounted_return"] = [
            [item["undiscounted_return"] for item in state["summaries"][horizon]]
            for state in states
        ]
    longest = MAX_CONTINUATION
    label_sets["success_ever"] = [
        [float(item["success_ever"]) for item in state["summaries"][longest]]
        for state in states
    ]
    label_sets["time_to_success"] = [
        [
            -float(
                item["first_success_step"]
                if item["first_success_step"] is not None
                else CHUNK_HORIZON + longest + 1
            )
            for item in state["summaries"][longest]
        ]
        for state in states
    ]
    for key in (
        "final_progress",
        "max_progress",
        "progress_transition_count",
        "potential_delta",
    ):
        label_sets[key] = [
            [float(item[key]) for item in state["summaries"][longest]]
            for state in states
        ]
    thresholds["potential_delta"] = MEANINGFUL_POTENTIAL_DELTA

    terminal_states = [state for state in states if state["terminal"]]
    terminal_critics = [state["critic_scores"] for state in terminal_states]
    if terminal_states:
        label_sets["terminal_success"] = [
            [float(item["success"]) for item in state["terminal"]]
            for state in terminal_states
        ]
        label_sets["terminal_discounted_return"] = [
            [item["discounted_return"] for item in state["terminal"]]
            for state in terminal_states
        ]
        label_sets["terminal_undiscounted_return"] = [
            [item["undiscounted_return"] for item in state["terminal"]]
            for state in terminal_states
        ]

    rows = []
    for name, values in label_sets.items():
        row_critics = (
            terminal_critics if name.startswith("terminal_") else critics
        )
        rows.append(
            (
                name,
                label_metrics(
                    values,
                    row_critics,
                    meaningful_threshold=thresholds.get(name, 0.0),
                ),
            )
        )
    return rows


def format_number(value):
    return "nan" if not np.isfinite(value) else f"{value:.6f}"


def main():
    np.random.seed(DIAGNOSTIC_SEED)
    random.seed(DIAGNOSTIC_SEED)
    env = make_ogbench_env_and_datasets(
        ENV_NAME, dataset_dir=DATASET_DIR, env_only=True
    )
    try:
        install_deterministic_reset_action_space(env)
        observation, _ = deterministic_reset(env, EPISODES[0][0])
        agent = create_restored_agent(
            observation, np.zeros(env.action_space.shape, dtype=np.float32)
        )
        print(
            "configuration: states=20 candidates=8 chunk=5 "
            "continuations=[20,40,80] terminal_extra_states=2 "
            f"baseline_actor_num_samples={agent.config['actor_num_samples']}"
        )
        print(
            "reward_semantics: reward=partial_progress-3; "
            "partial_progress=sum(cube_distance<=0.04); "
            "success=all_three_cubes; terminate_at_goal=True"
        )

        states = []
        episode_summaries = []
        for episode_index, (seed, seed_hint) in enumerate(EPISODES):
            search_seeds = (
                FAILURE_SEARCH_SEEDS
                if episode_index == len(EPISODES) - 1
                else (seed,)
            )
            for seed in search_seeds:
                scan = scan_episode(env, agent, episode_index, seed)
                if episode_index != len(EPISODES) - 1 or not scan["success"]:
                    break
                print(
                    f"failure_search_skipped_success_seed={seed}",
                    flush=True,
                )
            role = (
                "baseline_success"
                if scan["success"]
                else "baseline_failure"
            )
            selected = select_records(scan)
            episode_summaries.append(
                (
                    seed,
                    role,
                    scan["steps"],
                    scan["success"],
                    selected,
                )
            )
            states.extend(
                replay_and_analyze(
                    env,
                    agent,
                    episode_index,
                    seed,
                    role,
                    scan,
                    selected,
                )
            )
        assert len(states) == 20

        print("sampled_states:")
        print(
            "id | seed | role | step | phase/reason | progress | "
            "potential | closest_distance | q_margin | terminal_extra"
        )
        for index, state in enumerate(states, start=1):
            print(
                f"{index:>2} | {state['episode_seed']} | "
                f"{state['episode_role']} | {state['step']:>4} | "
                f"{state['phase_reason']} | "
                f"{state['initial_progress']} | "
                f"{state['initial_potential']:.6f} | "
                f"{state['closest_distance']:.6f} | "
                f"{state['critic_margin']:.6f} | "
                f"{bool(state['terminal'])}"
            )

        print("baseline_episodes:")
        for seed, role, steps, success, selected in episode_summaries:
            selected_text = ",".join(
                f"{record.step}:{reason}" for record, reason in selected
            )
            print(
                f"seed={seed} role={role} steps={steps} success={success} "
                f"selected={selected_text}"
            )

        print("reward_value_diagnostics:")
        for horizon in CONTINUATION_HORIZONS:
            unique_values = sorted(
                {
                    float(value)
                    for state in states
                    for item in state["summaries"][horizon]
                    for value in item["unique_rewards"]
                }
            )
            reward_stds = [
                item["reward_std"]
                for state in states
                for item in state["summaries"][horizon]
            ]
            print(
                f"continuation={horizon} total={CHUNK_HORIZON + horizon} "
                f"unique_rewards={unique_values} "
                f"reward_std_mean={np.mean(reward_stds):.6f}"
            )

        rows = collect_label_table(states)
        print("label_informativeness:")
        print(
            "label | mean_spread | median_spread | tied | meaningful | "
            "spearman(n) | pair_acc(n) | top1_regret | "
            "true_best_q_pct | critic_best_label_pct"
        )
        for name, metrics in rows:
            print(
                f"{name} | {metrics['mean_spread']:.6f} | "
                f"{metrics['median_spread']:.6f} | "
                f"{metrics['tied_states']}/{metrics['state_count']} | "
                f"{metrics['meaningful_fraction']:.6f} | "
                f"{format_number(metrics['spearman'])}"
                f"({metrics['spearman_states']}) | "
                f"{format_number(metrics['pairwise_accuracy'])}"
                f"({metrics['pair_count']}) | "
                f"{metrics['top1_regret']:.6f} | "
                f"{metrics['true_best_critic_percentile']:.6f} | "
                f"{metrics['critic_best_label_percentile']:.6f}"
            )

        print(
            "main_environment_pollution_checks_passed: "
            f"{sum(state['pollution_passed'] for state in states)}/20"
        )
        print(
            "repeated_branch_checks_passed: "
            f"{sum(state['repeat_passed'] for state in states)}/20"
        )
        print("branch label informativeness diagnostic passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
