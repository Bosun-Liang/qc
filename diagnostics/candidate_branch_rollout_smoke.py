"""Tiny evaluation-only candidate branch rollout diagnostic.

This script does not alter training, online collection, or baseline action
selection.  It restores the QC best checkpoint, generates four candidates
through the existing actor flow and critic, and evaluates them from exact
MuJoCo snapshots.
"""

import copy
import json
import random
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from agents.acfql import ACFQLAgent, get_config
from envs.mujoco_state import (
    clone_mujoco_env_state,
    restore_mujoco_env_state,
)
from envs.ogbench_utils import make_ogbench_env_and_datasets
from utils.flax_utils import restore_agent_with_file


ENV_NAME = "cube-triple-play-singletask-task2-v0"
DATASET_DIR = "/root/.ogbench/data"
CHECKPOINT = (
    "/root/autodl-tmp/qc_workspace/experiments/"
    "qc_cube_triple_online_1m/qc/qc_cube_triple_online_1m/"
    "cube-triple-play-singletask-task2-v0/"
    "sd00020260726_165011/params_1900000.pkl"
)
FLAGS_FILE = (
    "/root/autodl-tmp/qc_workspace/experiments/"
    "qc_cube_triple_online_1m/qc/qc_cube_triple_online_1m/"
    "cube-triple-play-singletask-task2-v0/"
    "sd00020260726_165011/flags.json"
)

NUM_STATES = 5
NUM_CANDIDATES = 4
CHUNK_HORIZON = 5
CONTINUATION_HORIZON = 20
EVAL_SEED = 20260728
DIAGNOSTIC_SEED = 7301
POTENTIAL_TEMPERATURE = 0.04
SUCCESS_TOLERANCE = 0.04


@dataclass
class BranchResult:
    rewards: np.ndarray
    actions: np.ndarray
    chunk_discounted_return: float
    continuation_discounted_return: float
    total_discounted_return: float
    total_undiscounted_return: float
    success: bool
    terminated: bool
    truncated: bool
    executed_steps: int
    initial_observation: np.ndarray
    final_observation: np.ndarray
    initial_qpos: np.ndarray
    final_qpos: np.ndarray
    initial_qvel: np.ndarray
    final_qvel: np.ndarray
    potential_delta: float
    critic_score: float


def create_restored_agent(observation, action):
    """Recreate the checkpoint architecture from its saved flags."""
    with open(FLAGS_FILE) as file:
        saved_flags = json.load(file)

    config = get_config()
    for name, value in saved_flags["agent"].items():
        if value is not None and name in config:
            config[name] = value
    config["horizon_length"] = saved_flags["horizon_length"]
    config["discount"] = saved_flags["discount"]

    if config["actor_type"] != "best-of-n":
        raise ValueError("Checkpoint is not configured for best-of-N")
    if not config["action_chunking"]:
        raise ValueError("Checkpoint does not use action chunking")
    if config["horizon_length"] != CHUNK_HORIZON:
        raise ValueError("Checkpoint action-chunk horizon is not five")

    agent = ACFQLAgent.create(
        seed=saved_flags["seed"],
        ex_observations=np.asarray(observation),
        ex_actions=np.asarray(action),
        config=config,
    )
    agent = restore_agent_with_file(agent, CHECKPOINT)
    return agent


@jax.jit
def generate_candidates_and_scores(agent, observation, rng):
    """Use the exact best-of-N actor-flow and critic scoring path."""
    flat_action_dim = agent.config["action_dim"] * CHUNK_HORIZON
    noises = jax.random.normal(
        rng, (NUM_CANDIDATES, flat_action_dim)
    )
    candidate_observations = jnp.repeat(
        observation[None, :], NUM_CANDIDATES, axis=0
    )
    candidate_actions = jnp.clip(
        agent.compute_flow_actions(candidate_observations, noises), -1, 1
    )
    candidate_qs = agent.network.select("critic")(
        candidate_observations, actions=candidate_actions
    )
    if agent.config["q_agg"] == "min":
        critic_scores = candidate_qs.min(axis=0)
    else:
        critic_scores = candidate_qs.mean(axis=0)
    return (
        candidate_actions.reshape(
            NUM_CANDIDATES,
            CHUNK_HORIZON,
            agent.config["action_dim"],
        ),
        critic_scores,
    )


def split_policy_rng(policy_rng):
    """Return the persisted next key and the key for one policy call."""
    next_rng, call_rng = jax.random.split(policy_rng)
    return next_rng, call_rng


def sample_baseline_chunk(agent, observation, policy_rng):
    """Call the unchanged baseline sample_actions and expose its next key."""
    next_rng, call_rng = split_policy_rng(policy_rng)
    flat_actions = agent.sample_actions(
        observations=jnp.asarray(observation), rng=call_rng
    )
    action_chunk = np.asarray(flat_actions).reshape(
        CHUNK_HORIZON, agent.config["action_dim"]
    )
    return action_chunk, next_rng


def cube_potential(env):
    """Compute the same smooth cube-success potential used previously."""
    core = env.unwrapped
    distances = []
    for cube_index in range(core._num_cubes):
        cube_xyz = core.data.joint(
            f"object_joint_{cube_index}"
        ).qpos[:3]
        target_xyz = core.data.mocap_pos[
            core._cube_target_mocap_ids[cube_index]
        ]
        distances.append(np.linalg.norm(cube_xyz - target_xyz))
    distances = np.asarray(distances)
    return float(
        np.sum(
            1.0
            / (
                1.0
                + np.exp(
                    (distances - SUCCESS_TOLERANCE)
                    / POTENTIAL_TEMPERATURE
                )
            )
        )
    )


def run_branch(
    env,
    snapshot,
    initial_observation,
    candidate_chunk,
    critic_score,
    agent,
    continuation_rng,
):
    """Execute one 5-step candidate plus 20-step baseline continuation."""
    restore_mujoco_env_state(env, snapshot)
    observation = np.array(initial_observation, copy=True)
    initial_qpos = env.unwrapped.data.qpos.copy()
    initial_qvel = env.unwrapped.data.qvel.copy()
    initial_potential = cube_potential(env)

    rewards = []
    actions = []
    terminated = False
    truncated = False
    final_info = {}
    chunk_steps = 0

    for action in candidate_chunk:
        observation, reward, terminated, truncated, final_info = env.step(
            np.clip(action, -1, 1)
        )
        rewards.append(float(reward))
        actions.append(np.array(action, copy=True))
        chunk_steps += 1
        if terminated or truncated:
            break

    continuation_steps = 0
    local_rng = jnp.array(continuation_rng, copy=True)
    action_queue = []
    while (
        continuation_steps < CONTINUATION_HORIZON
        and not (terminated or truncated)
    ):
        # Match evaluation.py exactly: sample_actions is called on every
        # environment step, while newly sampled chunks are only enqueued when
        # the current action queue is empty. Calls while the queue is non-empty
        # still advance the external policy RNG.
        sampled_chunk, local_rng = sample_baseline_chunk(
            agent, observation, local_rng
        )
        if not action_queue:
            action_queue.extend(sampled_chunk)
        action = action_queue.pop(0)
        observation, reward, terminated, truncated, final_info = env.step(
            np.clip(action, -1, 1)
        )
        rewards.append(float(reward))
        actions.append(np.array(action, copy=True))
        continuation_steps += 1

    rewards = np.asarray(rewards, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float32)
    discount = float(agent.config["discount"])
    chunk_rewards = rewards[:chunk_steps]
    continuation_rewards = rewards[chunk_steps:]
    chunk_discounted_return = float(
        np.sum(
            np.power(discount, np.arange(len(chunk_rewards)))
            * chunk_rewards
        )
    )
    continuation_discounted_return = float(
        np.sum(
            np.power(discount, np.arange(len(continuation_rewards)))
            * continuation_rewards
        )
    )
    total_discounted_return = float(
        chunk_discounted_return
        + discount**chunk_steps * continuation_discounted_return
    )

    return BranchResult(
        rewards=rewards,
        actions=actions,
        chunk_discounted_return=chunk_discounted_return,
        continuation_discounted_return=continuation_discounted_return,
        total_discounted_return=total_discounted_return,
        total_undiscounted_return=float(np.sum(rewards)),
        success=bool(final_info.get("success", False)),
        terminated=bool(terminated),
        truncated=bool(truncated),
        executed_steps=len(rewards),
        initial_observation=np.array(initial_observation, copy=True),
        final_observation=np.array(observation, copy=True),
        initial_qpos=initial_qpos,
        final_qpos=env.unwrapped.data.qpos.copy(),
        initial_qvel=initial_qvel,
        final_qvel=env.unwrapped.data.qvel.copy(),
        potential_delta=cube_potential(env) - initial_potential,
        critic_score=float(critic_score),
    )


def assert_branch_equal(first, second):
    np.testing.assert_array_equal(first.rewards, second.rewards)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(
        first.final_observation, second.final_observation
    )
    np.testing.assert_array_equal(first.final_qpos, second.final_qpos)
    np.testing.assert_array_equal(first.final_qvel, second.final_qvel)
    assert first.chunk_discounted_return == second.chunk_discounted_return
    assert (
        first.continuation_discounted_return
        == second.continuation_discounted_return
    )
    assert first.total_discounted_return == second.total_discounted_return
    assert first.total_undiscounted_return == second.total_undiscounted_return
    assert first.success == second.success
    assert first.terminated == second.terminated
    assert first.truncated == second.truncated
    assert first.executed_steps == second.executed_steps
    assert first.potential_delta == second.potential_delta


def wrapper_counters(env):
    counters = []
    current = env
    seen = set()
    names = (
        "_elapsed_steps",
        "_has_reset",
        "checked_reset",
        "checked_step",
        "checked_render",
        "close_called",
        "reward_sum",
        "episode_length",
        "total_timesteps",
    )
    while id(current) not in seen:
        seen.add(id(current))
        counters.append(
            {
                name: copy.deepcopy(getattr(current, name))
                for name in names
                if hasattr(current, name)
            }
        )
        if not hasattr(current, "env"):
            break
        current = current.env
    return counters


def capture_control_transition(env, action, next_policy_rng):
    observation, reward, terminated, truncated, _ = env.step(action)
    return {
        "observation": np.array(observation, copy=True),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "qpos": env.unwrapped.data.qpos.copy(),
        "qvel": env.unwrapped.data.qvel.copy(),
        "wrapper_counters": wrapper_counters(env),
        "env_random": float(env.unwrapped.np_random.random()),
        "numpy_random": float(np.random.random()),
        "python_random": float(random.random()),
        "policy_rng": np.asarray(next_policy_rng),
    }


def assert_control_equal(control, actual):
    np.testing.assert_array_equal(
        control["observation"], actual["observation"]
    )
    assert control["reward"] == actual["reward"]
    assert control["terminated"] == actual["terminated"]
    assert control["truncated"] == actual["truncated"]
    np.testing.assert_array_equal(control["qpos"], actual["qpos"])
    np.testing.assert_array_equal(control["qvel"], actual["qvel"])
    assert control["wrapper_counters"] == actual["wrapper_counters"]
    assert control["env_random"] == actual["env_random"]
    assert control["numpy_random"] == actual["numpy_random"]
    assert control["python_random"] == actual["python_random"]
    np.testing.assert_array_equal(control["policy_rng"], actual["policy_rng"])


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left, right):
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def descending_ranking(values):
    """Format candidate IDs in descending groups, preserving exact ties."""
    values = np.asarray(values)
    order = np.argsort(-values, kind="stable")
    groups = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        groups.append("=".join(str(index + 1) for index in order[start:end]))
        start = end
    return "[" + ",".join(groups) + "]"


def main():
    np.random.seed(DIAGNOSTIC_SEED)
    random.seed(DIAGNOSTIC_SEED)
    env = make_ogbench_env_and_datasets(
        ENV_NAME, dataset_dir=DATASET_DIR, env_only=True
    )
    try:
        observation, _ = env.reset(seed=EVAL_SEED)
        agent = create_restored_agent(
            observation, np.zeros(env.action_space.shape, dtype=np.float32)
        )
        print(f"checkpoint: {CHECKPOINT}")
        print(
            "configuration: states=5 candidates=4 chunk=5 "
            "continuation=20 baseline_actor_num_samples="
            f"{agent.config['actor_num_samples']} discount="
            f"{agent.config['discount']}"
        )

        main_policy_rng = jax.random.PRNGKey(EVAL_SEED)
        diagnostic_rng = jax.random.PRNGKey(DIAGNOSTIC_SEED)
        state_rows = []
        pollution_checks = 0
        repeat_checks = 0

        for state_index in range(NUM_STATES):
            snapshot = clone_mujoco_env_state(env)
            snapshot_policy_rng = jnp.array(main_policy_rng, copy=True)
            initial_observation = np.array(observation, copy=True)

            candidate_rng = jax.random.fold_in(
                diagnostic_rng, state_index
            )
            candidate_chunks, critic_scores = (
                generate_candidates_and_scores(
                    agent, jnp.asarray(initial_observation), candidate_rng
                )
            )
            candidate_chunks = np.asarray(candidate_chunks)
            critic_scores = np.asarray(critic_scores)
            assert candidate_chunks.shape == (
                NUM_CANDIDATES,
                CHUNK_HORIZON,
                agent.config["action_dim"],
            )
            assert critic_scores.shape == (NUM_CANDIDATES,)

            # Build the no-branch control from the same env and policy state.
            baseline_chunk, control_next_rng = sample_baseline_chunk(
                agent, initial_observation, snapshot_policy_rng
            )
            control = capture_control_transition(
                env, baseline_chunk[0], control_next_rng
            )

            restore_mujoco_env_state(env, snapshot)
            main_policy_rng = jnp.array(snapshot_policy_rng, copy=True)
            continuation_rng = jax.random.fold_in(
                diagnostic_rng, 10_000 + state_index
            )

            branch_results = [
                run_branch(
                    env=env,
                    snapshot=snapshot,
                    initial_observation=initial_observation,
                    candidate_chunk=candidate_chunks[candidate_index],
                    critic_score=critic_scores[candidate_index],
                    agent=agent,
                    continuation_rng=continuation_rng,
                )
                for candidate_index in range(NUM_CANDIDATES)
            ]

            repeated = run_branch(
                env=env,
                snapshot=snapshot,
                initial_observation=initial_observation,
                candidate_chunk=candidate_chunks[0],
                critic_score=critic_scores[0],
                agent=agent,
                continuation_rng=continuation_rng,
            )
            assert_branch_equal(branch_results[0], repeated)
            repeat_checks += 1

            # Restore the main path, regenerate the unchanged baseline action,
            # and compare against the no-branch control bit-for-bit.
            restore_mujoco_env_state(env, snapshot)
            main_policy_rng = jnp.array(snapshot_policy_rng, copy=True)
            actual_baseline_chunk, main_policy_rng = sample_baseline_chunk(
                agent, initial_observation, main_policy_rng
            )
            np.testing.assert_array_equal(
                baseline_chunk, actual_baseline_chunk
            )
            actual = capture_control_transition(
                env, actual_baseline_chunk[0], main_policy_rng
            )
            assert_control_equal(control, actual)
            pollution_checks += 1

            # Finish the already-selected baseline chunk to reach the next
            # main-trajectory diagnostic state.
            observation = actual["observation"]
            main_done = actual["terminated"] or actual["truncated"]
            for action in actual_baseline_chunk[1:]:
                if main_done:
                    break
                # Match evaluation.py RNG consumption while the previously
                # selected action queue is non-empty.
                _, main_policy_rng = sample_baseline_chunk(
                    agent, observation, main_policy_rng
                )
                (
                    observation,
                    _,
                    terminated,
                    truncated,
                    _,
                ) = env.step(np.clip(action, -1, 1))
                main_done = terminated or truncated
            if main_done and state_index + 1 < NUM_STATES:
                observation, _ = env.reset(seed=EVAL_SEED + state_index + 1)

            returns = np.asarray(
                [
                    result.total_discounted_return
                    for result in branch_results
                ]
            )
            critic_top = int(np.argmax(critic_scores))
            true_top = int(np.argmax(returns))
            row = {
                "state": state_index + 1,
                "returns": returns,
                "spread": float(np.ptp(returns)),
                "critic_scores": critic_scores,
                "critic_ranking": descending_ranking(critic_scores),
                "true_ranking": descending_ranking(returns),
                "critic_top": critic_top + 1,
                "true_top": true_top + 1,
                "regret": float(np.max(returns) - returns[critic_top]),
                "spearman": spearman(critic_scores, returns),
                "all_equal": bool(np.ptp(returns) == 0),
                "branches": branch_results,
            }
            state_rows.append(row)

        print(f"candidate_action_tensor_shape: {candidate_chunks.shape}")
        print(f"main_environment_pollution_checks_passed: {pollution_checks}/5")
        print(f"repeated_branch_checks_passed: {repeat_checks}/5")
        print(
            "state | returns | spread | critic_scores | critic_rank | "
            "true_rank | critic_top | true_top | regret | spearman"
        )
        for row in state_rows:
            returns_text = np.array2string(
                row["returns"], precision=6, separator=","
            )
            critic_text = np.array2string(
                row["critic_scores"], precision=6, separator=","
            )
            spearman_text = (
                "nan"
                if np.isnan(row["spearman"])
                else f"{row['spearman']:.6f}"
            )
            print(
                f"{row['state']:>5} | {returns_text} | "
                f"{row['spread']:.6f} | {critic_text} | "
                f"{row['critic_ranking']} | {row['true_ranking']} | "
                f"{row['critic_top']} | {row['true_top']} | "
                f"{row['regret']:.6f} | {spearman_text}"
            )
        print(
            "states_with_all_candidate_returns_equal: "
            f"{sum(row['all_equal'] for row in state_rows)}/{NUM_STATES}"
        )
        print("candidate branch rollout smoke test passed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
