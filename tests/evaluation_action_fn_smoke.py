"""Smoke test for optional deterministic evaluation action functions."""

import jax.numpy as jnp
import numpy as np

from evaluation import evaluate


class DummyAgent:
    def sample_actions(self, observations, rng):
        del observations, rng
        return jnp.asarray([0.0])


class OneStepEnv:
    def __init__(self):
        self.reset_seeds = []
        self.actions = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(np.asarray(action))
        info = {"success": float(action[0] == 1.0)}
        return (
            np.asarray([0.0], dtype=np.float32),
            0.0,
            True,
            False,
            info,
        )


def scored_actions(observations, rng):
    del observations, rng
    return jnp.asarray([1.0])


def main():
    env = OneStepEnv()
    stats, trajectories, renders = evaluate(
        agent=DummyAgent(),
        env=env,
        num_eval_episodes=2,
        num_video_episodes=0,
        action_dim=1,
        sample_actions_fn=scored_actions,
        eval_seed=17,
    )

    assert env.reset_seeds == [17, 18]
    assert all(float(action[0]) == 1.0 for action in env.actions)
    assert float(stats["success"]) == 1.0
    assert len(trajectories) == 2
    assert renders == []
    print("evaluation action-function smoke test passed")


if __name__ == "__main__":
    main()
