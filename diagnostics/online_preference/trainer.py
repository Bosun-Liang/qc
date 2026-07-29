"""Offline-initialized online preference selector trainer."""

import json
from itertools import combinations
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diagnostics.low_capacity_selector import (
    create_low_capacity_state,
    normalized_state_inputs,
)
from diagnostics.pairwise_selector import (
    compute_normalization,
    load_and_validate_dataset,
    percentile,
    rank_correlation,
    restore_state,
    save_state,
)
from utils.flax_utils import TrainState


PAIR_INDICES_8 = np.asarray(list(combinations(range(8), 2)), dtype=np.int32)


def ranking_metrics_variable(returns, scores):
    pair_correct = 0.0
    pair_count = 0
    correlations = []
    regrets = []
    top1 = 0
    selected_percentiles = []
    score_spreads = []
    disagreements = 0
    selector_wins = 0
    critic_wins = 0
    ties = 0
    for labels, predictions in zip(returns, scores):
        labels = np.asarray(labels, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        correlation = rank_correlation(labels, predictions)
        if np.isfinite(correlation):
            correlations.append(correlation)
        selected = int(np.argmax(predictions))
        top1 += labels[selected] == np.max(labels)
        regrets.append(float(np.max(labels) - labels[selected]))
        selected_percentiles.append(percentile(labels[selected], labels))
        score_spreads.append(float(np.ptp(predictions)))
        for left, right in combinations(range(len(labels)), 2):
            label_delta = labels[left] - labels[right]
            if label_delta == 0:
                continue
            score_delta = predictions[left] - predictions[right]
            pair_correct += (
                0.5
                if score_delta == 0
                else float(np.sign(score_delta) == np.sign(label_delta))
            )
            pair_count += 1
    count = len(returns)
    return {
        "states": count,
        "pair_count": pair_count,
        "pairwise_accuracy": float(pair_correct / pair_count) if pair_count else None,
        "spearman": float(np.mean(correlations)) if correlations else None,
        "spearman_states": len(correlations),
        "top1_accuracy": float(top1 / count) if count else None,
        "top1_regret": float(np.mean(regrets)) if regrets else None,
        "selected_label_percentile": (
            float(np.mean(selected_percentiles)) if selected_percentiles else None
        ),
        "score_mean": float(np.mean(scores)) if count else None,
        "score_std": float(np.std(scores)) if count else None,
        "mean_score_spread": float(np.mean(score_spreads)) if count else None,
    }


def shadow_comparison(records, selector_scores):
    selector_returns = []
    critic_returns = []
    selector_wins = 0
    critic_wins = 0
    ties = 0
    disagreements = 0
    for record, scores in zip(records, selector_scores):
        labels = np.asarray(record["discounted_returns"])
        selector_index = int(np.argmax(scores))
        critic_index = int(record["critic_selected_local_index"])
        selector_return = float(labels[selector_index])
        critic_return = float(labels[critic_index])
        selector_returns.append(selector_return)
        critic_returns.append(critic_return)
        if selector_index != critic_index:
            disagreements += 1
            if selector_return > critic_return:
                selector_wins += 1
            elif selector_return < critic_return:
                critic_wins += 1
            else:
                ties += 1
    count = len(records)
    return {
        "states": count,
        "disagreement_states": disagreements,
        "disagreement_rate": disagreements / count if count else None,
        "selector_win_rate_on_disagreement": (
            selector_wins / disagreements if disagreements else None
        ),
        "critic_win_rate_on_disagreement": (
            critic_wins / disagreements if disagreements else None
        ),
        "tie_rate_on_disagreement": ties / disagreements if disagreements else None,
        "selector_regret": float(
            np.mean(
                [
                    np.max(record["discounted_returns"]) - value
                    for record, value in zip(records, selector_returns)
                ]
            )
        ) if count else None,
        "critic_regret": float(
            np.mean(
                [
                    np.max(record["discounted_returns"]) - value
                    for record, value in zip(records, critic_returns)
                ]
            )
        ) if count else None,
    }


def _pairwise_loss(params, apply_fn, observations, actions, returns, valid, epsilon):
    scores = apply_fn({"params": params}, observations, actions)
    left = jnp.asarray(PAIR_INDICES_8[:, 0])
    right = jnp.asarray(PAIR_INDICES_8[:, 1])
    deltas = returns[:, left] - returns[:, right]
    pair_valid = valid[:, left] & valid[:, right] & (jnp.abs(deltas) >= epsilon)
    logits = (scores[:, left] - scores[:, right]) * jnp.sign(deltas)
    count = jnp.maximum(pair_valid.sum(), 1)
    loss = (jax.nn.softplus(-logits) * pair_valid).sum() / count
    correct = jnp.where(logits > 0, 1.0, jnp.where(logits == 0, 0.5, 0.0))
    valid_count = jnp.maximum(valid.sum(), 1)
    score_mean = (scores * valid).sum() / valid_count
    score_variance = (((scores - score_mean) ** 2) * valid).sum() / valid_count
    return loss, {
        "loss": loss,
        "pairwise_accuracy": (correct * pair_valid).sum() / count,
        "pair_count": pair_valid.sum(),
        "score_mean": score_mean,
        "score_std": jnp.sqrt(score_variance),
    }


@jax.jit
def online_update_step(state, observations, actions, returns, valid, epsilon):
    def loss_fn(params):
        return _pairwise_loss(
            params,
            state.apply_fn,
            observations,
            actions,
            returns,
            valid,
            epsilon,
        )

    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.asarray(
        [jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(grads)]
    ).all()
    return state.apply_gradients(grads=grads), {
        **metrics,
        "gradient_norm": optax.global_norm(grads),
        "gradient_finite": finite,
    }


class OnlineSelectorTrainer:
    def __init__(
        self,
        checkpoint,
        normalization_path,
        offline_dataset,
        output_dir,
        seed=0,
        pair_epsilon=1.0,
        learning_rate=3e-4,
        weight_decay=1e-4,
        gradient_clip=1.0,
    ):
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "selector_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = np.random.default_rng(seed + 80_080)
        self.pair_epsilon = float(pair_epsilon)
        base_template = create_low_capacity_state("small_mlp", seed)
        restored = restore_state(checkpoint, base_template)
        second = restore_state(checkpoint, create_low_capacity_state("small_mlp", seed))
        for left, right in zip(
            jax.tree_util.tree_leaves(restored.params),
            jax.tree_util.tree_leaves(second.params),
        ):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        tx = optax.chain(
            optax.clip_by_global_norm(gradient_clip),
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        )
        self.state = TrainState.create(restored.model_def, restored.params, tx=tx)
        with np.load(normalization_path, allow_pickle=False) as source:
            self.normalization = {
                key: np.array(source[key], copy=True)
                for key in (
                    "observation_mean",
                    "observation_std",
                    "action_mean",
                    "action_std",
                )
            }
        if self.normalization["observation_mean"].shape != (46,):
            raise ValueError("Offline observation normalization shape mismatch")
        if self.normalization["action_mean"].shape != (25,):
            raise ValueError("Offline action normalization shape mismatch")
        _, grouped = load_and_validate_dataset(offline_dataset)
        computed = compute_normalization(grouped["train"])
        np.testing.assert_array_equal(
            self.normalization["observation_mean"], computed.observation_mean
        )
        np.testing.assert_array_equal(
            self.normalization["observation_std"], computed.observation_std
        )
        np.testing.assert_array_equal(
            self.normalization["action_mean"], computed.action_mean
        )
        np.testing.assert_array_equal(
            self.normalization["action_std"], computed.action_std
        )
        self.offline = grouped["train"]
        offline_observations, offline_actions = normalized_state_inputs(
            self.offline, computed
        )
        self.offline_observations = offline_observations
        self.offline_actions = offline_actions
        fixed_predictions = np.asarray(
            self.state(offline_observations[:1], offline_actions[:1])
        )
        second_predictions = np.asarray(
            second(offline_observations[:1], offline_actions[:1])
        )
        np.testing.assert_array_equal(fixed_predictions, second_predictions)
        self.initial_fixed_prediction = fixed_predictions
        self.update_index = 0

    def normalized_online(self, records):
        count = len(records)
        observations = np.stack([record["observation"] for record in records])
        actions4 = np.stack([record["candidate_action_chunks"] for record in records]).reshape(count, 4, 25)
        observation_z = (
            observations - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]
        action_z = (
            actions4 - self.normalization["action_mean"]
        ) / self.normalization["action_std"]
        repeated_observations = np.repeat(observation_z[:, None, :], 4, axis=1)
        return repeated_observations.astype(np.float32), action_z.astype(np.float32)

    def predict_records(self, records):
        if not records:
            return np.zeros((0, 4), dtype=np.float32)
        observations, actions = self.normalized_online(records)
        return np.asarray(self.state(observations, actions))

    def ood_statistics(self, observation, candidate_chunks):
        observation_z = (
            observation - self.normalization["observation_mean"]
        ) / self.normalization["observation_std"]
        action_z = (
            candidate_chunks.reshape(4, 25) - self.normalization["action_mean"]
        ) / self.normalization["action_std"]
        values = np.concatenate([observation_z.reshape(-1), action_z.reshape(-1)])
        return {
            "normalized_abs_max": float(np.max(np.abs(values))),
            "fraction_abs_gt_3": float(np.mean(np.abs(values) > 3)),
            "fraction_abs_gt_5": float(np.mean(np.abs(values) > 5)),
            "is_ood": bool(np.any(np.abs(values) > 5)),
            "observation_normalized_abs_max": np.abs(observation_z).astype(np.float32),
            "action_normalized_abs_max": np.max(np.abs(action_z), axis=0).astype(np.float32),
        }

    def _mixed_batch(self, online_records, batch_states):
        desired_online = int(round(0.75 * batch_states))
        online_count = min(desired_online, len(online_records))
        offline_count = batch_states - online_count
        offline_indices = self.rng.choice(
            len(self.offline.state_ids), size=offline_count, replace=False
        )
        observations = [self.offline_observations[offline_indices]]
        actions = [self.offline_actions[offline_indices]]
        returns = [self.offline.returns[offline_indices]]
        valid = [np.ones((offline_count, 8), dtype=bool)]
        if online_count:
            selected = self.rng.choice(
                len(online_records), size=online_count, replace=False
            )
            records = [online_records[index] for index in selected]
            online_observations, online_actions4 = self.normalized_online(records)
            padded_observations = np.zeros((online_count, 8, 46), dtype=np.float32)
            padded_actions = np.zeros((online_count, 8, 25), dtype=np.float32)
            padded_returns = np.zeros((online_count, 8), dtype=np.float32)
            padded_valid = np.zeros((online_count, 8), dtype=bool)
            padded_observations[:, :4] = online_observations
            padded_actions[:, :4] = online_actions4
            padded_returns[:, :4] = np.stack(
                [record["discounted_returns"] for record in records]
            )
            padded_valid[:, :4] = True
            observations.append(padded_observations)
            actions.append(padded_actions)
            returns.append(padded_returns)
            valid.append(padded_valid)
        return (
            np.concatenate(observations),
            np.concatenate(actions),
            np.concatenate(returns),
            np.concatenate(valid),
            online_count,
            offline_count,
        )

    def update(self, online_records, gradient_steps, batch_states, main_env_step):
        if not online_records:
            raise ValueError("Online update requires train records")
        started = __import__("time").time()
        metrics = []
        online_examples = 0
        offline_examples = 0
        for _ in range(gradient_steps):
            batch = self._mixed_batch(online_records, batch_states)
            observations, actions, returns, valid, online_count, offline_count = batch
            self.state, info = online_update_step(
                self.state,
                observations,
                actions,
                returns,
                valid,
                np.float32(self.pair_epsilon),
            )
            info = {key: np.asarray(value).item() for key, value in info.items()}
            if not info["gradient_finite"] or not np.isfinite(info["loss"]):
                raise FloatingPointError("Non-finite online selector update")
            metrics.append(info)
            online_examples += online_count
            offline_examples += offline_count
        jax.tree_util.tree_map(
            lambda value: value.block_until_ready(), self.state.params
        )
        self.update_index += 1
        checkpoint = self.checkpoint_dir / f"update_{self.update_index:04d}.pkl"
        save_state(checkpoint, self.state)
        elapsed = __import__("time").time() - started
        total_examples = online_examples + offline_examples
        return {
            "selector_update_index": self.update_index,
            "main_env_step": main_env_step,
            "gradient_steps": gradient_steps,
            "train_loss": float(np.mean([value["loss"] for value in metrics])),
            "train_pairwise_accuracy": float(
                np.mean([value["pairwise_accuracy"] for value in metrics])
            ),
            "gradient_norm": float(
                np.mean([value["gradient_norm"] for value in metrics])
            ),
            "score_mean": float(np.mean([value["score_mean"] for value in metrics])),
            "score_std": float(np.mean([value["score_std"] for value in metrics])),
            "online_batch_fraction": online_examples / total_examples,
            "offline_batch_fraction": offline_examples / total_examples,
            "online_state_reuse_per_update": online_examples / len(online_records),
            "wall_clock_seconds": elapsed,
            "checkpoint": str(checkpoint),
        }
