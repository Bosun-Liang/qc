"""Episode-grouped state replay for online preference shadow labels."""

import json
from pathlib import Path

import numpy as np


class OnlinePreferenceBuffer:
    """A small resumable buffer whose atomic item is one candidate state."""

    def __init__(self, directory, holdout_modulus=5, holdout_remainder=0, resume=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.npz_path = self.directory / "online_states.npz"
        self.metadata_path = self.directory / "metadata.json"
        self.log_path = self.directory / "collection_log.jsonl"
        self.holdout_modulus = int(holdout_modulus)
        self.holdout_remainder = int(holdout_remainder)
        if not 0 <= self.holdout_remainder < self.holdout_modulus:
            raise ValueError("holdout_remainder must be within the modulus")
        self.records = []
        self.next_state_id = 0
        if resume:
            self.restore()

    @staticmethod
    def _validate(record):
        expected = {
            "observation": (46,),
            "candidate_action_chunks": (4, 5, 5),
            "critic_scores": (4,),
            "all_critic_scores": (32,),
            "top4_indices": (4,),
            "discounted_returns": (4,),
            "undiscounted_returns": (4,),
            "reward_sequences": (4, 85),
            "reward_masks": (4, 85),
            "success_ever": (4,),
            "first_success_step": (4,),
            "initial_progress": (4,),
            "final_progress": (4,),
            "max_progress": (4,),
            "potential_delta": (4,),
            "terminated": (4,),
            "truncated": (4,),
            "executed_steps": (4,),
            "selector_scores_at_collection": (4,),
        }
        for key, shape in expected.items():
            value = np.asarray(record[key])
            if value.shape != shape:
                raise ValueError(f"{key} shape {value.shape} != {shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")
        if not np.array_equal(
            np.asarray(record["reward_masks"]).sum(axis=1),
            np.asarray(record["executed_steps"]),
        ):
            raise ValueError("reward masks disagree with executed steps")

    def insert(self, record):
        record = dict(record)
        record["online_state_id"] = self.next_state_id
        record["source"] = "online"
        record["is_holdout"] = (
            int(record["episode_id"]) % self.holdout_modulus
            == self.holdout_remainder
        )
        record.setdefault("episode_success", False)
        record.setdefault("episode_complete", False)
        self._validate(record)
        self.records.append(record)
        self.next_state_id += 1
        with open(self.log_path, "a") as file:
            summary = {
                key: record[key]
                for key in (
                    "online_state_id",
                    "main_env_step",
                    "episode_id",
                    "state_step",
                    "is_holdout",
                    "critic_regret",
                    "selector_shadow_regret",
                    "critic_selector_disagreement",
                    "collection_timestamp_utc",
                )
            }
            file.write(json.dumps(summary, sort_keys=True) + "\n")
        return record["online_state_id"]

    def finalize_episode(self, episode_id, success):
        for record in self.records:
            if int(record["episode_id"]) == int(episode_id):
                record["episode_success"] = bool(success)
                record["episode_complete"] = True

    @property
    def train_records(self):
        return [record for record in self.records if not record["is_holdout"]]

    @property
    def holdout_records(self):
        return [record for record in self.records if record["is_holdout"]]

    def save(self, extra_metadata=None):
        if self.records:
            keys = self.records[0].keys()
            arrays = {}
            for key in keys:
                values = [record[key] for record in self.records]
                arrays[key] = (
                    np.stack(values)
                    if isinstance(values[0], np.ndarray)
                    else np.asarray(values)
                )
            np.savez_compressed(self.npz_path, **arrays)
        metadata = {
            "states": len(self.records),
            "train_states": len(self.train_records),
            "holdout_states": len(self.holdout_records),
            "train_episodes": sorted(
                {int(record["episode_id"]) for record in self.train_records}
            ),
            "holdout_episodes": sorted(
                {int(record["episode_id"]) for record in self.holdout_records}
            ),
            "holdout_rule": f"episode_id % {self.holdout_modulus} == {self.holdout_remainder}",
            "next_state_id": self.next_state_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        with open(self.metadata_path, "w") as file:
            json.dump(metadata, file, indent=2, sort_keys=True)

    def restore(self):
        if not self.npz_path.exists():
            return
        with np.load(self.npz_path, allow_pickle=False) as source:
            arrays = {key: np.array(source[key], copy=True) for key in source.files}
        row_count = len(arrays["online_state_id"])
        self.records = []
        for index in range(row_count):
            record = {key: value[index] for key, value in arrays.items()}
            for key, value in list(record.items()):
                if isinstance(value, np.generic):
                    record[key] = value.item()
            self._validate(record)
            self.records.append(record)
        self.next_state_id = (
            max(int(record["online_state_id"]) for record in self.records) + 1
            if self.records
            else 0
        )
        train_episodes = {
            int(record["episode_id"]) for record in self.train_records
        }
        holdout_episodes = {
            int(record["episode_id"]) for record in self.holdout_records
        }
        if not train_episodes.isdisjoint(holdout_episodes):
            raise ValueError("Episode leakage between online train and holdout")
