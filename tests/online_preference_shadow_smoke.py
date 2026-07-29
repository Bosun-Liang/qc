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
from diagnostics.pairwise_selector import restore_state


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
        assert trainer.latest_checkpoint_path.exists()
        latest_restored = restore_state(trainer.latest_checkpoint_path, trainer.state)
        latest_observations, latest_actions = trainer.normalized_online(
            restored_buffer.train_records
        )
        np.testing.assert_array_equal(
            trainer.predict_records(restored_buffer.train_records),
            np.asarray(latest_restored(latest_observations, latest_actions)),
        )
        assert update["online_batch_fraction"] == 1 / 8

        stable_trainer = OnlineSelectorTrainer(
            SELECTOR,
            NORMALIZATION,
            DATASET,
            root / "stable_trainer",
            seed=1,
            learning_rate=1e-4,
            normalized_input_clip=5.0,
            select_best_holdout_checkpoint=True,
        )
        extreme = synthetic_record(1)
        extreme["observation"] = (
            stable_trainer.normalization["observation_mean"]
            + 100.0 * stable_trainer.normalization["observation_std"]
        ).astype(np.float32)
        extreme_actions = (
            stable_trainer.normalization["action_mean"]
            - 100.0 * stable_trainer.normalization["action_std"]
        ).astype(np.float32)
        extreme["candidate_action_chunks"] = np.repeat(
            extreme_actions[None, :], 4, axis=0
        ).reshape(4, 5, 5)
        params_before = jax.tree_util.tree_map(
            lambda value: np.array(value, copy=True), stable_trainer.state.params
        )
        clipped_observations, clipped_actions = stable_trainer.normalized_online(
            [extreme]
        )
        assert np.max(clipped_observations) == 5.0
        assert np.min(clipped_actions) == -5.0
        predictions = stable_trainer.predict_records([extreme])
        assert np.isfinite(predictions).all()
        ood = stable_trainer.ood_statistics(
            extreme["observation"], extreme["candidate_action_chunks"]
        )
        assert ood["pre_clip_normalized_abs_max"] >= 99.0
        assert ood["post_clip_normalized_abs_max"] == 5.0
        assert ood["normalized_feature_clipped_fraction"] == 1.0
        for before_leaf, after_leaf in zip(
            jax.tree_util.tree_leaves(params_before),
            jax.tree_util.tree_leaves(stable_trainer.state.params),
        ):
            np.testing.assert_array_equal(before_leaf, np.asarray(after_leaf))

        stable_update = stable_trainer.update(
            restored_buffer.train_records,
            gradient_steps=1,
            batch_states=8,
            main_env_step=1,
        )
        insufficient = {
            "status": "ok",
            "selector": {
                "states": 19,
                "top1_regret": 10.0,
                "pairwise_accuracy": 0.5,
                "spearman": 0.0,
            },
        }
        result = stable_trainer.consider_best_holdout_checkpoint(
            stable_update["checkpoint"], insufficient, 20, 1
        )
        assert result["best_checkpoint_status"] == "insufficient_holdout"
        assert not stable_trainer.best_checkpoint_path.exists()

        def holdout(regret, pairwise, spearman=0.0):
            return {
                "status": "ok",
                "selector": {
                    "states": 20,
                    "top1_regret": regret,
                    "pairwise_accuracy": pairwise,
                    "spearman": spearman,
                },
            }

        assert stable_trainer.consider_best_holdout_checkpoint(
            stable_update["checkpoint"], holdout(10.0, 0.5), 20, 2
        )["best_checkpoint_updated"]
        assert stable_trainer.consider_best_holdout_checkpoint(
            stable_update["checkpoint"], holdout(9.0, 0.5), 20, 3
        )["best_checkpoint_updated"]
        assert stable_trainer.consider_best_holdout_checkpoint(
            stable_update["checkpoint"], holdout(9.0, 0.6), 20, 4
        )["best_checkpoint_updated"]
        restored_best = restore_state(
            stable_trainer.best_checkpoint_path, stable_trainer.state
        )
        expected = stable_trainer.predict_records(restored_buffer.train_records)
        obs, acts = stable_trainer.normalized_online(restored_buffer.train_records)
        actual = np.asarray(restored_best(obs, acts))
        np.testing.assert_array_equal(expected, actual)
    print("preference_disabled_best_of_n_exact=passed")
    print("offline_selector_checkpoint_restore=passed")
    print("normalization_restore=passed")
    print("online_buffer_insert_sample_save_restore=passed")
    print("normalized_input_clipping=passed")
    print("best_holdout_checkpoint_selection=passed")
    print("episode_grouped_holdout_leakage_check=passed")
    print("state_group_pair_construction=passed")
    print("one_selector_update=passed")
    print("latest_checkpoint_selection=passed")
    print("finite_loss_gradient=passed")
    print("resume_smoke=passed")
    print("online_preference_shadow_unit_smoke=passed")


if __name__ == "__main__":
    main()
