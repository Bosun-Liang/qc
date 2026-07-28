"""Deterministic clone/restore smoke test on the real OGBench cube env."""

import random

import numpy as np

from envs.mujoco_state import (
    clone_mujoco_env_state,
    restore_mujoco_env_state,
)
from envs.ogbench_utils import make_ogbench_env_and_datasets


ENV_NAME = "cube-triple-play-singletask-task2-v0"
DATASET_DIR = "/root/.ogbench/data"


def assert_exact_transition(first, second):
    first_ob, first_reward, first_terminated, first_truncated, _ = first
    second_ob, second_reward, second_terminated, second_truncated, _ = second
    np.testing.assert_array_equal(first_ob, second_ob)
    assert first_reward == second_reward
    assert first_terminated == second_terminated
    assert first_truncated == second_truncated


def main():
    env = make_ogbench_env_and_datasets(
        ENV_NAME,
        dataset_dir=DATASET_DIR,
        env_only=True,
    )
    try:
        env.reset(seed=20260728)
        warmup_action = np.asarray(
            [0.15, -0.10, 0.05, 0.20, -0.25], dtype=np.float32
        )
        env.step(warmup_action)

        snapshot = clone_mujoco_env_state(env)
        snapshot_qpos = env.unwrapped.data.qpos.copy()
        snapshot_qvel = env.unwrapped.data.qvel.copy()
        snapshot_elapsed_steps = env._elapsed_steps

        action = np.asarray(
            [-0.30, 0.20, -0.10, 0.35, 0.40], dtype=np.float32
        )

        # Deliberately consume every captured RNG family between branches.
        first = env.step(action)
        first_qpos = env.unwrapped.data.qpos.copy()
        first_qvel = env.unwrapped.data.qvel.copy()
        first_env_random = env.unwrapped.np_random.random()
        first_numpy_random = np.random.random()
        first_python_random = random.random()

        restore_mujoco_env_state(env, snapshot)
        np.testing.assert_array_equal(env.unwrapped.data.qpos, snapshot_qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, snapshot_qvel)
        assert env._elapsed_steps == snapshot_elapsed_steps

        second = env.step(action)
        second_qpos = env.unwrapped.data.qpos.copy()
        second_qvel = env.unwrapped.data.qvel.copy()
        second_env_random = env.unwrapped.np_random.random()
        second_numpy_random = np.random.random()
        second_python_random = random.random()

        assert_exact_transition(first, second)
        np.testing.assert_array_equal(first_qpos, second_qpos)
        np.testing.assert_array_equal(first_qvel, second_qvel)
        assert first_env_random == second_env_random
        assert first_numpy_random == second_numpy_random
        assert first_python_random == second_python_random

        # Leave the main environment exactly at the pre-branch state.
        restore_mujoco_env_state(env, snapshot)
        np.testing.assert_array_equal(env.unwrapped.data.qpos, snapshot_qpos)
        np.testing.assert_array_equal(env.unwrapped.data.qvel, snapshot_qvel)
        assert env._elapsed_steps == snapshot_elapsed_steps

        print("MuJoCo state clone/restore smoke test passed")
        print(
            "exactly matched: observation, reward, qpos, qvel, "
            "terminated, truncated, and RNG streams"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
