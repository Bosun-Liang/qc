"""Unit smoke tests for the online preference shadow pipeline."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.candidate_branch_rollout_smoke import (
    CHECKPOINT,
    create_restored_agent,
)
from diagnostics.online_preference.buffer import OnlinePreferenceBuffer
from diagnostics.online_preference.trainer import OnlineSelectorTrainer


DATASET = "/root/autodl-tmp/qc_workspace/datasets/qc_branch_preferences_h80_seed0.npz"
SELECTOR = "/root/autodl-tmp/qc_workspace/experiments/pairwise_selector_low_capacity_seed0/small_mlp/n140/best.pkl"
NORMALIZATION = "/root/autodl-tmp/qc_workspace/experiments/pairwise_selector_low_capacity_seed0/small_mlp/n140/normalization.npz"


@jax.jit
def legacy_best_of_n(agent, observation, rng):
    action_dim = agent.config["action_dim"] * agent.config["horizon_length"]
    noises = jax.random.normal(
        rng, (agent.config["actor_num_samples"], action_dim)
    )
    observations = jnp.repeat(
        observation[None, :], agent.config["actor_num_samples"], axis=0
    )
    actions = jnp.clip(
        agent.compute_flow_actions(observations, noises), -1, 1
    )
    candidate_qs = agent.network.select("critic")(observations, actions)
    scores = (
        candidate_qs.mean(axis=0)
        if agent.config["q_agg"] == "mean"
        else candidate_qs.min(axis=0)
    )
    index = jnp.argmax(scores)
    return actions, scores, index, actions[index]


def synthetic_record(episode_id, offset=0.0):
    rewards = np.full((4, 85), -3.0, dtype=np.float32)
    masks = np.ones((4, 85), dtype=bool)
    returns = np.asarray([-100.0, -98.0, -96.0, -94.0]) + offset
    return {
        "main_env_step": 1,
        "episode_id": episode_id,
        "episode_seed": -1,
        "state_step": 0,
        "state_phase": "early",
        "observation": np.zeros(46, dtype=np.float32),
        "candidate_action_chunks": np.zeros((4, 5, 5), dtype=np.float32),
        "critic_scores": np.asarray([4, 3, 2, 1], dtype=np.float64),
        "all_critic_scores": np.arange(32, dtype=np.float64),
        "top4_indices": np.asarray([31, 30, 29, 28], dtype=np.int32),
        "discounted_returns": returns,
        "undiscounted_returns": returns,
        "reward_sequences": rewards,
        "reward_masks": masks,
        "success_ever": np.zeros(4, dtype=bool),
        "first_success_step": np.full(4, -1, dtype=np.int32),
        "initial_progress": np.zeros(4, dtype=np.int32),
        "final_progress": np.zeros(4, dtype=np.int32),
        "max_progress": np.zeros(4, dtype=np.int32),
        "potential_delta": np.zeros(4, dtype=np.float64),
        "terminated": np.zeros(4, dtype=bool),
        "truncated": np.zeros(4, dtype=bool),
        "executed_steps": np.full(4, 85, dtype=np.int32),
        "critic_selected_local_index": 0,
        "selector_selected_local_index_at_collection": 0,
        "true_best_local_indices": np.asarray([3, -1, -1, -1], dtype=np.int32),
        "critic_regret": 6.0,
        "selector_shadow_regret": 6.0,
        "critic_selector_disagreement": False,
        "selector_scores_at_collection": np.zeros(4, dtype=np.float32),
        "critic_margin": 1.0,
        "normalized_abs_max": 0.0,
        "fraction_abs_gt_3": 0.0,
        "fraction_abs_gt_5": 0.0,
        "is_ood": False,
        "observation_normalized_abs": np.zeros(46, dtype=np.float32),
        "action_normalized_abs_max": np.zeros(25, dtype=np.float32),
        "collection_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "current_agent_update_step": 0,
        "current_checkpoint_identifier": CHECKPOINT,
    }


def main():
    observation = np.zeros(46, dtype=np.float32)
    action = np.zeros(5, dtype=np.float32)
    agent = create_restored_agent(observation, action)
    key = jax.random.PRNGKey(123)
    legacy_candidates, legacy_scores, legacy_index, legacy_action = legacy_best_of_n(
        agent, jnp.asarray(observation), key
    )
    candidates, scores, index = agent.sample_best_of_n_candidates(
        jnp.asarray(observation), key
    )
    selected = agent.sample_actions(jnp.asarray(observation), key)
    np.testing.assert_array_equal(np.asarray(candidates), np.asarray(legacy_candidates))
    np.testing.assert_array_equal(np.asarray(scores), np.asarray(legacy_scores))
    np.testing.assert_array_equal(np.asarray(index), np.asarray(legacy_index))
    np.testing.assert_array_equal(np.asarray(selected), np.asarray(legacy_action))

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        buffer = OnlinePreferenceBuffer(root / "buffer", holdout_modulus=5)
        buffer.insert(synthetic_record(0))
        buffer.insert(synthetic_record(1, 1.0))
        buffer.finalize_episode(0, True)
        buffer.finalize_episode(1, False)
        buffer.save()
        assert len(buffer.holdout_records) == 1
        assert len(buffer.train_records) == 1
        restored_buffer = OnlinePreferenceBuffer(
            root / "buffer", holdout_modulus=5, resume=True
        )
        assert len(restored_buffer.records) == 2
        assert {
            int(record["episode_id"]) for record in restored_buffer.train_records
        }.isdisjoint(
            {
                int(record["episode_id"])
                for record in restored_buffer.holdout_records
            }
        )

        trainer = OnlineSelectorTrainer(
            SELECTOR,
            NORMALIZATION,
            DATASET,
            root / "trainer",
            seed=0,
        )
        before = trainer.predict_records(restored_buffer.train_records)
        assert before.shape == (1, 4)
        update = trainer.update(
            restored_buffer.train_records,
            gradient_steps=1,
            batch_states=8,
            main_env_step=1,
        )
        assert np.isfinite(update["train_loss"])
        assert np.isfinite(update["gradient_norm"])
        assert Path(update["checkpoint"]).exists()
        assert update["online_batch_fraction"] == 1 / 8
    print("preference_disabled_best_of_n_exact=passed")
    print("offline_selector_checkpoint_restore=passed")
    print("normalization_restore=passed")
    print("online_buffer_insert_sample_save_restore=passed")
    print("episode_grouped_holdout_leakage_check=passed")
    print("state_group_pair_construction=passed")
    print("one_selector_update=passed")
    print("finite_loss_gradient=passed")
    print("resume_smoke=passed")
    print("online_preference_shadow_unit_smoke=passed")


if __name__ == "__main__":
    main()
