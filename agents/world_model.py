"""Latent action-chunk dynamics model and independent training state.

This module is intentionally independent from the policy and critic networks.
The first-stage world model only predicts the latent observation at the end of
an action chunk and does not participate in action selection.
"""

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP


class ObservationEncoder(nn.Module):
    """Encode state observations into a learned latent space."""

    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations):
        return MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
        )(observations)


class LatentDynamics(nn.Module):
    """Predict a future latent from a latent and a flattened action chunk."""

    latent_dim: int
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, latents, action_chunks):
        inputs = jnp.concatenate((latents, action_chunks), axis=-1)
        return MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
        )(inputs)


class LatentValueHead(nn.Module):
    """Predict a scalar future critic value from a predicted latent."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, latents):
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
        )(latents).squeeze(-1)


class WorldModel(nn.Module):
    """Encode observations and predict the latent after an action chunk."""

    latent_dim: int = 64
    hidden_dims: Sequence[int] = (256, 256)

    def setup(self):
        self.encoder = ObservationEncoder(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
        )
        self.dynamics = LatentDynamics(
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
        )

    def __call__(self, observations, actions, target_observations):
        latents = self.encoder(observations)
        target_latents = self.encoder(target_observations)
        flat_actions = actions.reshape((*actions.shape[:-2], -1))
        predicted_latents = self.dynamics(latents, flat_actions)
        return predicted_latents, target_latents


def world_model_loss(apply_fn, params, batch):
    """Compute a terminal-validity-masked latent dynamics loss.

    `batch["observations"]` is o_t and the last element of
    `batch["next_observations"]` is o_{t+H}. Chunks that cross a terminal
    boundary are excluded with `batch["valid"][..., -1]`.
    """

    target_observations = batch["next_observations"][..., -1, :]
    predicted_latents, target_latents = apply_fn(
        {"params": params},
        batch["observations"],
        batch["actions"],
        target_observations,
    )
    target_latents = jax.lax.stop_gradient(target_latents)

    per_sample_loss = jnp.mean(
        jnp.square(predicted_latents - target_latents),
        axis=-1,
    )
    valid = batch["valid"][..., -1].astype(per_sample_loss.dtype)
    valid_count = jnp.maximum(jnp.sum(valid), jnp.asarray(1.0, valid.dtype))
    loss = jnp.sum(per_sample_loss * valid) / valid_count

    flat_targets = target_latents.reshape((-1, target_latents.shape[-1]))
    flat_valid = valid.reshape((-1, 1))
    latent_mean = jnp.sum(flat_targets * flat_valid, axis=0) / valid_count
    latent_variance = jnp.sum(
        jnp.square(flat_targets - latent_mean) * flat_valid,
        axis=0,
    ) / valid_count
    pred_norm = jnp.linalg.norm(predicted_latents, axis=-1)
    target_norm = jnp.linalg.norm(target_latents, axis=-1)
    prediction_cosine = jnp.sum(
        predicted_latents * target_latents,
        axis=-1,
    ) / (
        jnp.linalg.norm(predicted_latents, axis=-1)
        * jnp.linalg.norm(target_latents, axis=-1)
    ).clip(1e-8)

    info = {
        "loss": loss,
        "pred_norm": jnp.sum(pred_norm * valid) / valid_count,
        "target_norm": jnp.sum(target_norm * valid) / valid_count,
        "latent_std": jnp.sqrt(jnp.maximum(latent_variance, 0.0)).mean(),
        "prediction_cosine": jnp.sum(prediction_cosine * valid) / valid_count,
        "valid_fraction": valid.mean(),
        "is_finite": jnp.logical_and(
            jnp.isfinite(loss),
            jnp.logical_and(
                jnp.all(jnp.isfinite(predicted_latents)),
                jnp.all(jnp.isfinite(target_latents)),
            ),
        ),
    }
    return loss, info


class WorldModelTrainState(flax.struct.PyTreeNode):
    """Independent optimizer state for auxiliary world-model training."""

    network: Any
    coef: float = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        example_observations,
        example_actions,
        horizon_length,
        latent_dim=64,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=0.1,
    ):
        model = WorldModel(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
        )
        action_chunk = jnp.stack(
            [example_actions] * horizon_length,
            axis=-2,
        )
        params = model.init(
            jax.random.PRNGKey(seed),
            example_observations,
            action_chunk,
            example_observations,
        )["params"]
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adam(learning_rate=learning_rate),
        )
        return cls(network=network, coef=coef)

    @staticmethod
    def _update(state, batch):
        def loss_fn(params):
            loss, info = world_model_loss(
                state.network.apply_fn,
                params,
                batch,
            )
            weighted_loss = state.coef * loss
            info["weighted_loss"] = weighted_loss
            return weighted_loss, info

        network, info = state.network.apply_loss_fn(loss_fn)
        return state.replace(network=network), info

    @jax.jit
    def update(self, batch):
        """Apply one world-model optimizer update."""
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Apply one update per leading UTD axis entry."""
        state, infos = jax.lax.scan(self._update, self, batch)
        infos = jax.tree_util.tree_map(lambda value: value.mean(), infos)
        return state, infos


class LatentValueTrainState(flax.struct.PyTreeNode):
    """Independent diagnostic head for future-value prediction.

    The head consumes a stopped-gradient world-model prediction. Its target is
    the stopped target-critic value at the real observation after the chunk.
    It never participates in policy action selection.
    """

    rng: Any
    network: Any
    coef: float = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        latent_dim=64,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=1.0,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        model = LatentValueHead(hidden_dims=hidden_dims)
        params = model.init(
            init_rng,
            jnp.zeros((latent_dim,), dtype=jnp.float32),
        )["params"]
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adam(learning_rate=learning_rate),
        )
        return cls(rng=rng, network=network, coef=coef)

    def value_loss(self, batch, world_model, agent, rng, params):
        target_observations = batch["next_observations"][..., -1, :]
        predicted_latents, _ = world_model.network(
            batch["observations"],
            batch["actions"],
            target_observations,
        )
        predicted_latents = jax.lax.stop_gradient(predicted_latents)

        next_actions = agent.sample_actions(target_observations, rng=rng)
        target_qs = agent.network.select("target_critic")(
            target_observations,
            actions=next_actions,
        )
        if agent.config["q_agg"] == "min":
            raw_target_values = target_qs.min(axis=0)
        else:
            raw_target_values = target_qs.mean(axis=0)
        raw_target_values = jax.lax.stop_gradient(raw_target_values)

        predictions = self.network(predicted_latents, params=params)
        valid = batch["valid"][..., -1].astype(predictions.dtype)
        valid_count = jnp.maximum(jnp.sum(valid), jnp.asarray(1.0, valid.dtype))
        raw_target_mean = jnp.sum(raw_target_values * valid) / valid_count
        raw_target_variance = jnp.sum(
            jnp.square(raw_target_values - raw_target_mean) * valid
        ) / valid_count
        raw_target_std = jnp.sqrt(jnp.maximum(raw_target_variance, 0.0))
        target_values = (
            raw_target_values - raw_target_mean
        ) / raw_target_std.clip(1e-6)
        errors = predictions - target_values
        loss = jnp.sum(jnp.square(errors) * valid) / valid_count
        mae = jnp.sum(jnp.abs(errors) * valid) / valid_count

        prediction_mean = jnp.sum(predictions * valid) / valid_count
        target_mean = jnp.sum(target_values * valid) / valid_count
        centered_predictions = predictions - prediction_mean
        centered_targets = target_values - target_mean
        covariance = jnp.sum(
            centered_predictions * centered_targets * valid
        ) / valid_count
        prediction_variance = jnp.sum(
            jnp.square(centered_predictions) * valid
        ) / valid_count
        target_variance = jnp.sum(
            jnp.square(centered_targets) * valid
        ) / valid_count
        correlation = covariance / jnp.sqrt(
            prediction_variance * target_variance
        ).clip(1e-8)

        pair_valid = valid[:-1] * valid[1:]
        pair_count = jnp.maximum(
            jnp.sum(pair_valid),
            jnp.asarray(1.0, pair_valid.dtype),
        )
        pairwise_agreement = jnp.sum(
            (
                (predictions[:-1] - predictions[1:])
                * (target_values[:-1] - target_values[1:])
                > 0
            ).astype(predictions.dtype)
            * pair_valid
        ) / pair_count

        topk = max(1, predictions.shape[0] // 10)
        masked_predictions = jnp.where(valid > 0, predictions, -jnp.inf)
        masked_targets = jnp.where(valid > 0, target_values, -jnp.inf)
        _, prediction_topk = jax.lax.top_k(masked_predictions, topk)
        _, target_topk = jax.lax.top_k(masked_targets, topk)
        topk_agreement = jnp.mean(
            jnp.isin(prediction_topk, target_topk).astype(predictions.dtype)
        )

        info = {
            "loss": loss,
            "mae": mae,
            "correlation": correlation,
            "pairwise_agreement": pairwise_agreement,
            "topk_agreement": topk_agreement,
            "prediction_mean": prediction_mean,
            "prediction_std": jnp.sqrt(jnp.maximum(prediction_variance, 0.0)),
            "target_mean": target_mean,
            "target_std": jnp.sqrt(jnp.maximum(target_variance, 0.0)),
            "raw_target_mean": raw_target_mean,
            "raw_target_std": raw_target_std,
            "valid_fraction": valid.mean(),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(predictions)),
                jnp.logical_and(
                    jnp.logical_and(
                        jnp.all(jnp.isfinite(raw_target_values)),
                        jnp.all(jnp.isfinite(target_values)),
                    ),
                    jnp.all(jnp.isfinite(jnp.asarray((loss, correlation)))),
                ),
            ),
        }
        return loss, info

    @staticmethod
    def _update(state, batch, world_model, agent):
        new_rng, sample_rng = jax.random.split(state.rng)

        def loss_fn(params):
            loss, info = state.value_loss(
                batch,
                world_model,
                agent,
                sample_rng,
                params,
            )
            weighted_loss = state.coef * loss
            info["weighted_loss"] = weighted_loss
            return weighted_loss, info

        network, info = state.network.apply_loss_fn(loss_fn)
        return state.replace(rng=new_rng, network=network), info

    @jax.jit
    def update(self, batch, world_model, agent):
        """Apply one diagnostic value-head update."""
        return self._update(self, batch, world_model, agent)

    @jax.jit
    def batch_update(self, batch, world_model, agent):
        """Apply one value-head update per leading UTD axis entry."""

        def scan_update(state, scan_batch):
            return self._update(state, scan_batch, world_model, agent)

        state, infos = jax.lax.scan(scan_update, self, batch)
        infos = jax.tree_util.tree_map(lambda value: value.mean(), infos)
        return state, infos

    @jax.jit
    def evaluate(self, batch, world_model, agent):
        """Evaluate diagnostics with a fixed RNG and without changing state."""
        _, info = self.value_loss(
            batch,
            world_model,
            agent,
            jax.random.PRNGKey(0),
            self.network.params,
        )
        return info

    @jax.jit
    def evaluate_candidates(self, observations, world_model, agent, rng):
        """Compare critic and latent-value rankings on identical candidates.

        This is a read-only diagnostic. It reproduces best-of-N candidate
        generation but does not return an action or affect action selection.
        """
        num_samples = agent.config["actor_num_samples"]
        horizon_length = agent.config["horizon_length"]
        action_dim = agent.config["action_dim"]
        flat_action_dim = action_dim * horizon_length

        noises = jax.random.normal(
            rng,
            (*observations.shape[:-1], num_samples, flat_action_dim),
        )
        candidate_observations = jnp.repeat(
            observations[..., None, :], num_samples, axis=-2
        )
        candidate_actions = agent.compute_flow_actions(
            candidate_observations, noises
        )
        candidate_actions = jnp.clip(candidate_actions, -1, 1)

        candidate_qs = agent.network.select("critic")(
            candidate_observations, actions=candidate_actions
        )
        if agent.config["q_agg"] == "min":
            critic_scores = candidate_qs.min(axis=0)
        else:
            critic_scores = candidate_qs.mean(axis=0)

        action_chunks = candidate_actions.reshape(
            (*candidate_actions.shape[:-1], horizon_length, action_dim)
        )
        predicted_latents, _ = world_model.network(
            candidate_observations,
            action_chunks,
            candidate_observations,
        )
        predicted_latents = jax.lax.stop_gradient(predicted_latents)
        value_scores = self.network(predicted_latents)

        critic_ranks = jnp.argsort(
            jnp.argsort(critic_scores, axis=-1), axis=-1
        ).astype(value_scores.dtype)
        value_ranks = jnp.argsort(
            jnp.argsort(value_scores, axis=-1), axis=-1
        ).astype(value_scores.dtype)

        def mean_correlation(left, right):
            left = left - left.mean(axis=-1, keepdims=True)
            right = right - right.mean(axis=-1, keepdims=True)
            numerator = jnp.sum(left * right, axis=-1)
            denominator = jnp.sqrt(
                jnp.sum(jnp.square(left), axis=-1)
                * jnp.sum(jnp.square(right), axis=-1)
            ).clip(1e-8)
            return jnp.mean(numerator / denominator)

        critic_top1 = jnp.argmax(critic_scores, axis=-1)
        value_top1 = jnp.argmax(value_scores, axis=-1)
        top1_agreement = jnp.mean(
            (critic_top1 == value_top1).astype(value_scores.dtype)
        )

        topk = max(1, num_samples // 10)
        _, critic_topk = jax.lax.top_k(critic_scores, topk)
        _, value_topk = jax.lax.top_k(value_scores, topk)
        topk_overlap = jnp.mean(
            jax.vmap(
                lambda left, right: jnp.mean(
                    jnp.any(left[:, None] == right[None, :], axis=-1).astype(
                        value_scores.dtype
                    )
                )
            )(critic_topk, value_topk)
        )

        critic_choice_value_rank = jnp.take_along_axis(
            value_ranks, critic_top1[..., None], axis=-1
        ).squeeze(-1)
        value_choice_critic_rank = jnp.take_along_axis(
            critic_ranks, value_top1[..., None], axis=-1
        ).squeeze(-1)
        rank_denominator = jnp.asarray(
            max(1, num_samples - 1), dtype=value_scores.dtype
        )

        critic_selected_actions = jnp.take_along_axis(
            candidate_actions,
            critic_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        value_selected_actions = jnp.take_along_axis(
            candidate_actions,
            value_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)

        return {
            "spearman_correlation": mean_correlation(
                critic_ranks, value_ranks
            ),
            "score_correlation": mean_correlation(
                critic_scores, value_scores
            ),
            "top1_agreement": top1_agreement,
            "topk_overlap": topk_overlap,
            "random_top1_agreement": jnp.asarray(
                1.0 / num_samples, dtype=value_scores.dtype
            ),
            "random_topk_overlap": jnp.asarray(
                topk / num_samples, dtype=value_scores.dtype
            ),
            "critic_choice_value_percentile": jnp.mean(
                critic_choice_value_rank / rank_denominator
            ),
            "value_choice_critic_percentile": jnp.mean(
                value_choice_critic_rank / rank_denominator
            ),
            "selected_action_l2": jnp.mean(
                jnp.linalg.norm(
                    critic_selected_actions - value_selected_actions,
                    axis=-1,
                )
            ),
            "critic_score_std": jnp.mean(jnp.std(critic_scores, axis=-1)),
            "value_score_std": jnp.mean(jnp.std(value_scores, axis=-1)),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(critic_scores)),
                jnp.all(jnp.isfinite(value_scores)),
            ),
        }
