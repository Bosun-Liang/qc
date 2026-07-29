"""Online preference collection/training hook that never controls actions."""

import copy
import datetime
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from diagnostics.branch_label_informativeness import (
    assert_trace_equal,
    prefix_summary,
    rollout_candidate,
)
from diagnostics.candidate_branch_rollout_smoke import wrapper_counters
from diagnostics.online_preference.buffer import OnlinePreferenceBuffer
from diagnostics.online_preference.trainer import (
    OnlineSelectorTrainer,
    ranking_metrics_variable,
    shadow_comparison,
)
from diagnostics.pairwise_selector import sha256_file
from envs.mujoco_state import clone_mujoco_env_state, restore_mujoco_env_state


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _rng_fingerprint(env):
    payload = {
        "numpy": copy.deepcopy(np.random.get_state()),
        "python": copy.deepcopy(__import__("random").getstate()),
        "environment": copy.deepcopy(env.unwrapped.np_random.bit_generator.state),
    }
    return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)


def _environment_fingerprint(env):
    return {
        "qpos": env.unwrapped.data.qpos.copy(),
        "qvel": env.unwrapped.data.qvel.copy(),
        "wrapper_counters": wrapper_counters(env),
        "rng": _rng_fingerprint(env),
    }


def _assert_environment_fingerprint(left, right):
    np.testing.assert_array_equal(left["qpos"], right["qpos"])
    np.testing.assert_array_equal(left["qvel"], right["qvel"])
    assert left["wrapper_counters"] == right["wrapper_counters"]
    assert left["rng"] == right["rng"]


def _post_step_fingerprint(
    env, observation, reward, terminated, truncated, policy_rng
):
    return {
        **_environment_fingerprint(env),
        "observation": np.array(observation, copy=True),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "policy_rng": np.asarray(policy_rng),
    }


def _assert_post_step_equal(left, right):
    _assert_environment_fingerprint(left, right)
    np.testing.assert_array_equal(left["observation"], right["observation"])
    assert left["reward"] == right["reward"]
    assert left["terminated"] == right["terminated"]
    assert left["truncated"] == right["truncated"]
    np.testing.assert_array_equal(left["policy_rng"], right["policy_rng"])


class OnlinePreferenceShadow:
    """A default-off diagnostic hook; selector scores never leave this class."""

    def __init__(
        self,
        env,
        output_dir,
        selector_checkpoint,
        selector_normalization,
        offline_dataset,
        qc_checkpoint,
        environment_name,
        seed,
        collection_interval_chunks=20,
        top_k=4,
        continuation_horizon=80,
        discount=0.99,
        pair_epsilon=1.0,
        min_online_states=50,
        update_every_states=20,
        gradient_steps=200,
        batch_states=32,
        holdout_modulus=5,
        holdout_remainder=0,
        recent_holdout_min_states=20,
        resume=False,
    ):
        if top_k != 4 or continuation_horizon != 80:
            raise ValueError("The shadow smoke requires top_k=4 and H80")
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.output_dir / "online_dataset"
        self.buffer = OnlinePreferenceBuffer(
            self.dataset_dir, holdout_modulus=holdout_modulus, holdout_remainder=holdout_remainder, resume=resume
        )
        self.trainer = OnlineSelectorTrainer(
            selector_checkpoint,
            selector_normalization,
            offline_dataset,
            self.output_dir,
            seed=seed,
            pair_epsilon=pair_epsilon,
        )
        self.seed = seed
        self.collection_interval_chunks = int(collection_interval_chunks)
        self.top_k = top_k
        self.continuation_horizon = continuation_horizon
        self.discount = discount
        self.min_online_states = int(min_online_states)
        self.update_every_states = int(update_every_states)
        self.gradient_steps = int(gradient_steps)
        self.batch_states = int(batch_states)
        self.recent_holdout_min_states = int(recent_holdout_min_states)
        self.chunk_boundaries = 0
        self.main_env_steps = 0
        self.branch_env_steps = 0
        self.branch_collection_seconds = 0.0
        self.selector_update_seconds = 0.0
        self.repeatability_checks = 0
        self.pollution_checks = 0
        self.qc_replay_checks = 0
        self.pending_control = None
        self.last_update_train_states = 0
        self.started = time.time()
        self.update_history = []
        self.episode_summaries = []
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.qc_update_count = 0
        self.training_log = self.output_dir / "training_history.jsonl"
        self.qc_log = self.output_dir / "qc_training_history.jsonl"
        self.metadata = {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "environment": environment_name,
            "qc_restore_checkpoint": qc_checkpoint,
            "selector_restore_checkpoint": selector_checkpoint,
            "selector_normalization": selector_normalization,
            "offline_dataset": offline_dataset,
            "offline_dataset_sha256": sha256_file(offline_dataset),
            "selector_normalization_sha256": sha256_file(selector_normalization),
            "seed": seed,
            "jax_devices": [str(device) for device in jax.devices()],
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "flags": {
                "collection_interval_chunks": collection_interval_chunks,
                "top_k": top_k,
                "continuation_horizon": continuation_horizon,
                "discount": discount,
                "pair_epsilon": pair_epsilon,
                "min_online_states": min_online_states,
                "update_every_states": update_every_states,
                "gradient_steps": gradient_steps,
                "batch_states": batch_states,
                "holdout_modulus": holdout_modulus,
                "holdout_remainder": holdout_remainder,
                "recent_holdout_min_states": recent_holdout_min_states,
                "resume": resume,
            },
        }
        self.metadata["gpu_memory_initial"] = self._gpu_memory()
        if resume:
            checkpoints = sorted(
                (self.output_dir / "selector_checkpoints").glob("update_*.pkl")
            )
            if checkpoints:
                from diagnostics.pairwise_selector import restore_state

                self.trainer.state = restore_state(
                    checkpoints[-1], self.trainer.state
                )
                self.trainer.update_index = int(checkpoints[-1].stem.split("_")[-1])
                self.last_update_train_states = len(self.buffer.train_records)
        with open(self.output_dir / "config.json", "w") as file:
            json.dump(_json_ready(self.metadata), file, indent=2, sort_keys=True)

    @staticmethod
    def _gpu_memory():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            return output
        except Exception as error:
            return f"unavailable: {error}"

    def should_collect(self):
        return self.chunk_boundaries % self.collection_interval_chunks == 0

    def on_chunk_boundary(
        self,
        agent,
        observation,
        candidates,
        critic_scores,
        critic_selected_index,
        main_policy_rng,
        main_env_step,
        episode_id,
        state_step,
        qc_replay,
        qc_update_count,
    ):
        """Possibly label top-4; returned values cannot influence actions."""
        self.chunk_boundaries += 1
        if not self.should_collect():
            return None
        started = time.time()
        candidates = np.asarray(candidates).reshape(32, 5, 5)
        critic_scores = np.asarray(critic_scores).reshape(32)
        critic_selected_index = int(np.asarray(critic_selected_index))
        if critic_selected_index != int(np.argmax(critic_scores)):
            raise AssertionError("Critic selected index disagrees with argmax")
        top4_indices = np.argsort(critic_scores, kind="stable")[-4:][::-1]
        if critic_selected_index not in top4_indices:
            raise AssertionError("Critic top-1 missing from top-4")
        top4_chunks = candidates[top4_indices]
        top4_scores = critic_scores[top4_indices]
        critic_selected_local = int(
            np.flatnonzero(top4_indices == critic_selected_index)[0]
        )
        if critic_selected_local != 0:
            raise AssertionError("Sorted critic top-4 must put top-1 first")
        snapshot = clone_mujoco_env_state(self.env)
        pre_branch = _environment_fingerprint(self.env)
        replay_before = (qc_replay.size, qc_replay.pointer)
        check_index = len(self.buffer.records)
        if check_index < 10:
            control_observation, control_reward, control_terminated, control_truncated, _ = self.env.step(
                top4_chunks[critic_selected_local, 0]
            )
            self.pending_control = _post_step_fingerprint(
                self.env,
                control_observation,
                control_reward,
                control_terminated,
                control_truncated,
                main_policy_rng,
            )
            restore_mujoco_env_state(self.env, snapshot)
        continuation_key = jax.random.fold_in(
            jax.random.PRNGKey(self.seed + 900_000), self.buffer.next_state_id
        )
        traces = [
            rollout_candidate(
                self.env,
                snapshot,
                observation,
                top4_chunks[index],
                top4_scores[index],
                agent,
                jnp.array(continuation_key, copy=True),
                self.continuation_horizon,
            )
            for index in range(4)
        ]
        if check_index < 10:
            repeated = rollout_candidate(
                self.env,
                snapshot,
                observation,
                top4_chunks[0],
                top4_scores[0],
                agent,
                jnp.array(continuation_key, copy=True),
                self.continuation_horizon,
            )
            assert_trace_equal(traces[0], repeated)
            self.repeatability_checks += 1
        restore_mujoco_env_state(self.env, snapshot)
        _assert_environment_fingerprint(
            pre_branch, _environment_fingerprint(self.env)
        )
        if replay_before != (qc_replay.size, qc_replay.pointer):
            raise AssertionError("Branch rollout changed QC replay")
        self.qc_replay_checks += 1
        summaries = [prefix_summary(trace, self.continuation_horizon) for trace in traces]
        rewards = np.zeros((4, 85), dtype=np.float32)
        masks = np.zeros((4, 85), dtype=bool)
        for index, summary in enumerate(summaries):
            sequence = summary["reward_sequence"]
            rewards[index, : len(sequence)] = sequence
            masks[index, : len(sequence)] = True
        labels = np.asarray(
            [summary["discounted_return"] for summary in summaries],
            dtype=np.float64,
        )
        selector_scores = self.trainer.predict_records(
            [
                {
                    "observation": np.asarray(observation),
                    "candidate_action_chunks": top4_chunks,
                }
            ]
        )[0]
        selector_index = int(np.argmax(selector_scores))
        true_best_mask = labels == np.max(labels)
        true_best_indices = np.full(4, -1, dtype=np.int32)
        true_indices = np.flatnonzero(true_best_mask)
        true_best_indices[: len(true_indices)] = true_indices
        ood = self.trainer.ood_statistics(observation, top4_chunks)
        record = {
            "main_env_step": int(main_env_step),
            "episode_id": int(episode_id),
            "episode_seed": -1,
            "state_step": int(state_step),
            "state_phase": (
                "early" if state_step < 333 else "middle" if state_step < 667 else "late"
            ),
            "observation": np.asarray(observation, dtype=np.float32),
            "candidate_action_chunks": top4_chunks.astype(np.float32),
            "critic_scores": top4_scores.astype(np.float64),
            "all_critic_scores": critic_scores.astype(np.float64),
            "top4_indices": top4_indices.astype(np.int32),
            "discounted_returns": labels,
            "undiscounted_returns": np.asarray(
                [summary["undiscounted_return"] for summary in summaries]
            ),
            "reward_sequences": rewards,
            "reward_masks": masks,
            "success_ever": np.asarray([summary["success_ever"] for summary in summaries]),
            "first_success_step": np.asarray([
                -1 if summary["first_success_step"] is None else summary["first_success_step"]
                for summary in summaries
            ], dtype=np.int32),
            "initial_progress": np.asarray([summary["initial_progress"] for summary in summaries], dtype=np.int32),
            "final_progress": np.asarray([summary["final_progress"] for summary in summaries], dtype=np.int32),
            "max_progress": np.asarray([summary["max_progress"] for summary in summaries], dtype=np.int32),
            "potential_delta": np.asarray([summary["potential_delta"] for summary in summaries]),
            "terminated": np.asarray([summary["terminated"] for summary in summaries]),
            "truncated": np.asarray([summary["truncated"] for summary in summaries]),
            "executed_steps": np.asarray([summary["executed_steps"] for summary in summaries], dtype=np.int32),
            "critic_selected_local_index": critic_selected_local,
            "selector_selected_local_index_at_collection": selector_index,
            "true_best_local_indices": true_best_indices,
            "critic_regret": float(np.max(labels) - labels[critic_selected_local]),
            "selector_shadow_regret": float(np.max(labels) - labels[selector_index]),
            "critic_selector_disagreement": bool(selector_index != critic_selected_local),
            "selector_scores_at_collection": selector_scores.astype(np.float32),
            "critic_margin": float(top4_scores[0] - top4_scores[1]),
            "normalized_abs_max": ood["normalized_abs_max"],
            "fraction_abs_gt_3": ood["fraction_abs_gt_3"],
            "fraction_abs_gt_5": ood["fraction_abs_gt_5"],
            "is_ood": ood["is_ood"],
            "observation_normalized_abs": ood["observation_normalized_abs_max"],
            "action_normalized_abs_max": ood["action_normalized_abs_max"],
            "collection_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "current_agent_update_step": int(qc_update_count),
            "current_checkpoint_identifier": self.metadata["qc_restore_checkpoint"],
        }
        state_id = self.buffer.insert(record)
        self.branch_env_steps += sum(summary["executed_steps"] for summary in summaries)
        self.branch_collection_seconds += time.time() - started
        self.buffer.save({"branch_env_steps": self.branch_env_steps})
        self._maybe_update(main_env_step)
        return state_id

    def after_main_step(
        self,
        observation,
        reward,
        terminated,
        truncated,
        main_policy_rng,
        executed_candidate_index,
        critic_selected_candidate_index,
    ):
        if int(executed_candidate_index) != int(critic_selected_candidate_index):
            raise AssertionError("Selector changed the executed candidate")
        self.main_env_steps += 1
        self.current_episode_return += float(reward)
        self.current_episode_length += 1
        if self.pending_control is not None:
            actual = _post_step_fingerprint(
                self.env,
                observation,
                reward,
                terminated,
                truncated,
                main_policy_rng,
            )
            _assert_post_step_equal(self.pending_control, actual)
            self.pending_control = None
            self.pollution_checks += 1

    def end_episode(self, episode_id, success):
        self.buffer.finalize_episode(episode_id, success)
        self.episode_summaries.append(
            {
                "episode_id": int(episode_id),
                "return": self.current_episode_return,
                "length": self.current_episode_length,
                "success": bool(success),
            }
        )
        self.current_episode_return = 0.0
        self.current_episode_length = 0
        self.buffer.save({"branch_env_steps": self.branch_env_steps})

    def record_qc_update(self, main_env_step, replay_size, agent_info):
        self.qc_update_count += 1
        payload = {
            "main_env_step": int(main_env_step),
            "qc_update_count": self.qc_update_count,
            "online_replay_size": int(replay_size),
        }
        for key, value in agent_info.items():
            array = np.asarray(value)
            if array.size == 1 and np.isfinite(array).all():
                payload[key] = array.item()
        with open(self.qc_log, "a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

    def _evaluate_records(self, records):
        if not records:
            return {"status": "insufficient", "states": 0}
        scores = self.trainer.predict_records(records)
        labels = np.stack([record["discounted_returns"] for record in records])
        selector = ranking_metrics_variable(labels, scores)
        critic_scores = np.stack([record["critic_scores"] for record in records])
        critic = ranking_metrics_variable(labels, critic_scores)
        return {
            "status": "ok",
            "selector": selector,
            "critic": critic,
            "comparison": shadow_comparison(records, scores),
        }

    def _maybe_update(self, main_env_step):
        train_records = self.buffer.train_records
        if len(train_records) < self.min_online_states:
            return
        if len(train_records) - self.last_update_train_states < self.update_every_states:
            return
        update = self.trainer.update(
            train_records,
            self.gradient_steps,
            self.batch_states,
            main_env_step,
        )
        self.selector_update_seconds += update["wall_clock_seconds"]
        self.last_update_train_states = len(train_records)
        holdout = self.buffer.holdout_records
        if len(holdout) >= self.recent_holdout_min_states:
            update["recent_holdout"] = self._evaluate_records(holdout)
        else:
            update["recent_holdout"] = {
                "status": "insufficient",
                "states": len(holdout),
                "required": self.recent_holdout_min_states,
            }
        update["online_train_states"] = len(train_records)
        update["online_holdout_states"] = len(holdout)
        self.update_history.append(update)
        with open(self.training_log, "a") as file:
            file.write(json.dumps(_json_ready(update), sort_keys=True) + "\n")

    def finalize(self, requested_main_steps):
        records = self.buffer.records
        overall = self._evaluate_records(records)
        holdout = self._evaluate_records(self.buffer.holdout_records)
        train = self._evaluate_records(self.buffer.train_records)
        margins = np.asarray([record["critic_margin"] for record in records]) if records else np.asarray([])
        subgroups = {}
        if records:
            median_margin = float(np.median(margins))
            groups = {
                "early": [record for record in records if record["state_phase"] == "early"],
                "middle": [record for record in records if record["state_phase"] == "middle"],
                "late": [record for record in records if record["state_phase"] == "late"],
                "progress_0": [record for record in records if record["initial_progress"][0] == 0],
                "progress_1": [record for record in records if record["initial_progress"][0] == 1],
                "progress_2": [record for record in records if record["initial_progress"][0] == 2],
                "episode_success": [record for record in records if record["episode_complete"] and record["episode_success"]],
                "episode_failure": [record for record in records if record["episode_complete"] and not record["episode_success"]],
                "high_critic_margin": [record for record in records if record["critic_margin"] >= median_margin],
                "low_critic_margin": [record for record in records if record["critic_margin"] < median_margin],
                "ood": [record for record in records if record["is_ood"]],
                "non_ood": [record for record in records if not record["is_ood"]],
            }
            subgroups = {name: self._evaluate_records(values) for name, values in groups.items()}
        elapsed = time.time() - self.started
        final = {
            "requested_main_env_steps": int(requested_main_steps),
            "main_env_steps": self.main_env_steps,
            "branch_env_steps": self.branch_env_steps,
            "total_effective_env_steps": self.main_env_steps + self.branch_env_steps,
            "labeled_states": len(records),
            "online_train_states": len(self.buffer.train_records),
            "online_holdout_states": len(self.buffer.holdout_records),
            "selector_updates": len(self.update_history),
            "qc_updates": self.qc_update_count,
            "repeatability_checks": self.repeatability_checks,
            "pollution_checks": self.pollution_checks,
            "qc_replay_unchanged_checks": self.qc_replay_checks,
            "executed_action_assertions": self.main_env_steps,
            "wall_clock_seconds": elapsed,
            "branch_collection_seconds": self.branch_collection_seconds,
            "selector_update_seconds": self.selector_update_seconds,
            "branch_seconds_per_labeled_state": self.branch_collection_seconds / len(records) if records else None,
            "selector_update_seconds_average": self.selector_update_seconds / len(self.update_history) if self.update_history else None,
            "episodes": self.episode_summaries,
            "overall": overall,
            "online_train": train,
            "recent_holdout": holdout,
            "subgroups": subgroups,
            "update_history": self.update_history,
            "gpu_memory_initial": self.metadata["gpu_memory_initial"],
            "gpu_memory_final": self._gpu_memory(),
            "normalization_ood": {
                "mean_fraction_abs_gt_3": float(np.mean([record["fraction_abs_gt_3"] for record in records])) if records else None,
                "mean_fraction_abs_gt_5": float(np.mean([record["fraction_abs_gt_5"] for record in records])) if records else None,
                "max_normalized_abs": float(np.max([record["normalized_abs_max"] for record in records])) if records else None,
                "ood_states": int(sum(record["is_ood"] for record in records)),
            },
        }
        with open(self.output_dir / "final_summary.json", "w") as file:
            json.dump(_json_ready(final), file, indent=2, sort_keys=True)
        with open(self.output_dir / "metrics.json", "w") as file:
            json.dump(_json_ready({"final": final}), file, indent=2, sort_keys=True)
        metadata = {**self.metadata, **{key: final[key] for key in (
            "main_env_steps", "branch_env_steps", "total_effective_env_steps", "wall_clock_seconds"
        )}}
        with open(self.output_dir / "metadata.json", "w") as file:
            json.dump(_json_ready(metadata), file, indent=2, sort_keys=True)
        self.buffer.save({"branch_env_steps": self.branch_env_steps})
        return final
