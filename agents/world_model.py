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


class LatentProgressHead(nn.Module):
    """Classify task progress from an observation latent."""

    hidden_dims: Sequence[int]
    num_classes: int = 4

    @nn.compact
    def __call__(self, latents):
        return MLP(
            (*self.hidden_dims, self.num_classes),
            activate_final=False,
        )(latents)


class LatentPotentialHead(nn.Module):
    """Regress a continuous task potential from an observation latent."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, latents):
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
        )(latents).squeeze(-1)


class LatentPotentialDeltaHead(nn.Module):
    """Predict potential change from a latent world-model transition."""

    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, latents, predicted_latents, action_chunks):
        flat_actions = action_chunks.reshape(
            (*action_chunks.shape[:-2], -1)
        )
        inputs = jnp.concatenate(
            (latents, predicted_latents - latents, flat_actions),
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
        )(inputs).squeeze(-1)


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

    def encode(self, observations):
        """Encode observations without running chunk dynamics."""
        return self.encoder(observations)

    def predict(self, observations, actions):
        """Predict an action-chunk future latent without a target encode."""
        latents = self.encoder(observations)
        flat_actions = actions.reshape((*actions.shape[:-2], -1))
        return self.dynamics(latents, flat_actions)


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

        info = {
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

        critic_score_std = jnp.std(
            critic_scores, axis=-1, keepdims=True
        ).clip(1e-6)
        value_score_std = jnp.std(
            value_scores, axis=-1, keepdims=True
        ).clip(1e-6)
        normalized_critic_scores = (
            critic_scores - critic_scores.mean(axis=-1, keepdims=True)
        ) / critic_score_std
        normalized_value_scores = (
            value_scores - value_scores.mean(axis=-1, keepdims=True)
        ) / value_score_std

        for score_lambda, label in (
            (0.0, "0"),
            (0.05, "0p05"),
            (0.1, "0p1"),
            (0.25, "0p25"),
            (0.5, "0p5"),
            (1.0, "1"),
        ):
            mixed_scores = (
                normalized_critic_scores
                + score_lambda * normalized_value_scores
            )
            mixed_top1 = jnp.argmax(mixed_scores, axis=-1)
            mixed_critic_rank = jnp.take_along_axis(
                critic_ranks, mixed_top1[..., None], axis=-1
            ).squeeze(-1)
            mixed_value_rank = jnp.take_along_axis(
                value_ranks, mixed_top1[..., None], axis=-1
            ).squeeze(-1)
            mixed_critic_score = jnp.take_along_axis(
                normalized_critic_scores,
                mixed_top1[..., None],
                axis=-1,
            ).squeeze(-1)
            mixed_value_score = jnp.take_along_axis(
                normalized_value_scores,
                mixed_top1[..., None],
                axis=-1,
            ).squeeze(-1)
            mixed_actions = jnp.take_along_axis(
                candidate_actions,
                mixed_top1[..., None, None],
                axis=-2,
            ).squeeze(-2)
            prefix = f"lambda_{label}"
            info[f"{prefix}/change_rate"] = jnp.mean(
                (mixed_top1 != critic_top1).astype(value_scores.dtype)
            )
            info[f"{prefix}/critic_percentile"] = jnp.mean(
                mixed_critic_rank / rank_denominator
            )
            info[f"{prefix}/value_percentile"] = jnp.mean(
                mixed_value_rank / rank_denominator
            )
            info[f"{prefix}/critic_regret_z"] = jnp.mean(
                jnp.max(normalized_critic_scores, axis=-1)
                - mixed_critic_score
            )
            info[f"{prefix}/value_regret_z"] = jnp.mean(
                jnp.max(normalized_value_scores, axis=-1)
                - mixed_value_score
            )
            info[f"{prefix}/value_top1_agreement"] = jnp.mean(
                (mixed_top1 == value_top1).astype(value_scores.dtype)
            )
            info[f"{prefix}/action_l2_from_critic"] = jnp.mean(
                jnp.linalg.norm(
                    mixed_actions - critic_selected_actions, axis=-1
                )
            )

        return info

    @jax.jit
    def sample_actions(
        self,
        observations,
        world_model,
        agent,
        rng,
        score_lambda=0.0,
    ):
        """Select best-of-N actions with an optional latent-value score.

        A zero coefficient follows the original critic-only argmax exactly.
        This method is kept separate from `ACFQLAgent.sample_actions` so the
        default policy and all training paths remain unchanged.
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
        value_scores = self.network(predicted_latents)

        normalized_critic_scores = (
            critic_scores - critic_scores.mean(axis=-1, keepdims=True)
        ) / jnp.std(critic_scores, axis=-1, keepdims=True).clip(1e-6)
        normalized_value_scores = (
            value_scores - value_scores.mean(axis=-1, keepdims=True)
        ) / jnp.std(value_scores, axis=-1, keepdims=True).clip(1e-6)
        mixed_scores = (
            normalized_critic_scores
            + score_lambda * normalized_value_scores
        )
        selection_scores = jax.lax.cond(
            jnp.asarray(score_lambda) == 0,
            lambda: critic_scores,
            lambda: mixed_scores,
        )
        indices = jnp.argmax(selection_scores, axis=-1)

        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        batch_size = len(flat_indices)
        return candidate_actions.reshape(
            (-1, num_samples, flat_action_dim)
        )[jnp.arange(batch_size), flat_indices, :].reshape(
            batch_shape + (flat_action_dim,)
        )

class LatentProgressTrainState(flax.struct.PyTreeNode):
    """Independent class-balanced task-progress diagnostic head.

    The head is supervised on real observation latents and never updates the
    world model, actor, or critic. Reward classes -3, -2, -1, and 0 map to
    progress classes 0, 1, 2, and 3 respectively.
    """

    network: Any
    coef: float = nonpytree_field()
    class_weights: Any = nonpytree_field()
    num_classes: int = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        latent_dim=64,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=1.0,
        class_weights=(1.0, 1.0, 1.0, 1.0),
        num_classes=4,
    ):
        if len(class_weights) != num_classes:
            raise ValueError(
                "class_weights length must match num_classes"
            )
        model = LatentProgressHead(
            hidden_dims=hidden_dims,
            num_classes=num_classes,
        )
        params = model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((latent_dim,), dtype=jnp.float32),
        )["params"]
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adam(learning_rate=learning_rate),
        )
        return cls(
            network=network,
            coef=coef,
            class_weights=tuple(float(x) for x in class_weights),
            num_classes=num_classes,
        )

    def progress_loss(self, batch, world_model, params):
        latents = world_model.network(
            batch["observations"], method="encode"
        )
        latents = jax.lax.stop_gradient(latents)
        logits = self.network(latents, params=params)

        rewards = batch["rewards"]
        if rewards.ndim > 1:
            rewards = rewards[..., 0]
        labels = jnp.clip(
            jnp.rint(rewards + 3.0).astype(jnp.int32),
            0,
            self.num_classes - 1,
        )
        if "valid" in batch:
            valid = batch["valid"]
            if valid.ndim > 1:
                valid = valid[..., 0]
            valid = valid.astype(logits.dtype)
        else:
            valid = jnp.ones(labels.shape, dtype=logits.dtype)

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        per_sample_loss = -jnp.take_along_axis(
            log_probs, labels[..., None], axis=-1
        ).squeeze(-1)
        class_weights = jnp.asarray(
            self.class_weights, dtype=logits.dtype
        )
        sample_weights = class_weights[labels] * valid
        weight_sum = jnp.maximum(
            jnp.sum(sample_weights),
            jnp.asarray(1.0, dtype=logits.dtype),
        )
        loss = jnp.sum(per_sample_loss * sample_weights) / weight_sum

        probabilities = jax.nn.softmax(logits, axis=-1)
        progress_values = jnp.arange(
            self.num_classes, dtype=logits.dtype
        )
        predicted_progress = jnp.sum(
            probabilities * progress_values, axis=-1
        )
        target_progress = labels.astype(logits.dtype)
        predictions = jnp.argmax(logits, axis=-1)
        valid_count = jnp.maximum(
            jnp.sum(valid),
            jnp.asarray(1.0, dtype=logits.dtype),
        )
        accuracy = jnp.sum(
            (predictions == labels).astype(logits.dtype) * valid
        ) / valid_count
        mae = jnp.sum(
            jnp.abs(predicted_progress - target_progress) * valid
        ) / valid_count

        prediction_mean = jnp.sum(predicted_progress * valid) / valid_count
        target_mean = jnp.sum(target_progress * valid) / valid_count
        centered_predictions = predicted_progress - prediction_mean
        centered_targets = target_progress - target_mean
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

        info = {
            "loss": loss,
            "accuracy": accuracy,
            "mae": mae,
            "correlation": correlation,
            "prediction_mean": prediction_mean,
            "prediction_std": jnp.sqrt(
                jnp.maximum(prediction_variance, 0.0)
            ),
            "target_mean": target_mean,
            "target_std": jnp.sqrt(jnp.maximum(target_variance, 0.0)),
            "valid_fraction": valid.mean(),
        }
        balanced_accuracy = jnp.asarray(0.0, dtype=logits.dtype)
        present_classes = jnp.asarray(0.0, dtype=logits.dtype)
        for class_index in range(self.num_classes):
            class_mask = (labels == class_index).astype(logits.dtype) * valid
            class_count = jnp.sum(class_mask)
            class_present = (class_count > 0).astype(logits.dtype)
            recall = jnp.sum(
                (predictions == class_index).astype(logits.dtype)
                * class_mask
            ) / jnp.maximum(
                class_count, jnp.asarray(1.0, dtype=logits.dtype)
            )
            info[f"class_{class_index}_fraction"] = (
                class_count / valid_count
            )
            info[f"class_{class_index}_recall"] = recall
            balanced_accuracy = (
                balanced_accuracy + recall * class_present
            )
            present_classes = present_classes + class_present
        info["balanced_accuracy"] = balanced_accuracy / jnp.maximum(
            present_classes, jnp.asarray(1.0, dtype=logits.dtype)
        )
        info["is_finite"] = jnp.logical_and(
            jnp.all(jnp.isfinite(logits)),
            jnp.all(
                jnp.isfinite(
                    jnp.asarray((loss, accuracy, mae, correlation))
                )
            ),
        )
        return loss, info

    @staticmethod
    def _update(state, batch, world_model):
        def loss_fn(params):
            loss, info = state.progress_loss(batch, world_model, params)
            weighted_loss = state.coef * loss
            info["weighted_loss"] = weighted_loss
            return weighted_loss, info

        network, info = state.network.apply_loss_fn(loss_fn)
        return state.replace(network=network), info

    @jax.jit
    def update(self, batch, world_model):
        """Apply one progress-head optimizer update."""
        return self._update(self, batch, world_model)

    @jax.jit
    def batch_update(self, batch, world_model):
        """Apply one progress-head update per leading UTD entry."""

        def scan_update(state, scan_batch):
            return self._update(state, scan_batch, world_model)

        state, infos = jax.lax.scan(scan_update, self, batch)
        infos = jax.tree_util.tree_map(lambda value: value.mean(), infos)
        return state, infos

    @jax.jit
    def evaluate(self, batch, world_model):
        """Evaluate task-progress metrics without changing state."""
        _, info = self.progress_loss(
            batch, world_model, self.network.params
        )
        return info

    @jax.jit
    def evaluate_predicted_future(self, batch, world_model):
        """Evaluate progress after composing dynamics with the head.

        The supplied action chunks and future observations come from held-out
        data. This is read-only and separates dynamics-to-head composition
        quality from classification on directly encoded observations.
        """
        predicted_latents, target_latents = world_model.network(
            batch["observations"],
            batch["actions"],
            batch["target_observations"],
        )
        predicted_latents = jax.lax.stop_gradient(predicted_latents)
        target_latents = jax.lax.stop_gradient(target_latents)
        predicted_logits = self.network(predicted_latents)
        encoded_logits = self.network(target_latents)
        predicted_probabilities = jax.nn.softmax(
            predicted_logits, axis=-1
        )
        encoded_probabilities = jax.nn.softmax(encoded_logits, axis=-1)
        progress_values = jnp.arange(
            self.num_classes, dtype=predicted_logits.dtype
        )
        predicted_scores = jnp.sum(
            predicted_probabilities * progress_values, axis=-1
        )
        encoded_scores = jnp.sum(
            encoded_probabilities * progress_values, axis=-1
        )
        labels = jnp.clip(
            jnp.rint(batch["target_rewards"] + 3.0).astype(jnp.int32),
            0,
            self.num_classes - 1,
        )
        targets = labels.astype(predicted_scores.dtype)
        predictions = jnp.argmax(predicted_logits, axis=-1)

        def correlation(left, right):
            left = left - left.mean()
            right = right - right.mean()
            return jnp.sum(left * right) / jnp.sqrt(
                jnp.sum(jnp.square(left)) * jnp.sum(jnp.square(right))
            ).clip(1e-8)

        info = {
            "accuracy": jnp.mean((predictions == labels).astype(
                predicted_scores.dtype
            )),
            "mae": jnp.mean(jnp.abs(predicted_scores - targets)),
            "correlation": correlation(predicted_scores, targets),
            "prediction_mean": jnp.mean(predicted_scores),
            "prediction_std": jnp.std(predicted_scores),
            "encoded_mean": jnp.mean(encoded_scores),
            "encoded_std": jnp.std(encoded_scores),
            "score_vs_encoded_mae": jnp.mean(
                jnp.abs(predicted_scores - encoded_scores)
            ),
            "score_vs_encoded_correlation": correlation(
                predicted_scores, encoded_scores
            ),
            "latent_mse": jnp.mean(
                jnp.square(predicted_latents - target_latents)
            ),
            "latent_cosine": jnp.mean(
                jnp.sum(predicted_latents * target_latents, axis=-1)
                / (
                    jnp.linalg.norm(predicted_latents, axis=-1)
                    * jnp.linalg.norm(target_latents, axis=-1)
                ).clip(1e-8)
            ),
        }
        balanced_accuracy = jnp.asarray(
            0.0, dtype=predicted_scores.dtype
        )
        present_classes = jnp.asarray(
            0.0, dtype=predicted_scores.dtype
        )
        for class_index in range(self.num_classes):
            class_mask = labels == class_index
            class_count = jnp.sum(class_mask)
            class_present = (class_count > 0).astype(
                predicted_scores.dtype
            )
            recall = jnp.sum(
                (predictions == class_index).astype(predicted_scores.dtype)
                * class_mask.astype(predicted_scores.dtype)
            ) / jnp.maximum(
                class_count,
                jnp.asarray(1, dtype=class_count.dtype),
            )
            info[f"class_{class_index}_recall"] = recall
            balanced_accuracy += recall * class_present
            present_classes += class_present
        info["balanced_accuracy"] = balanced_accuracy / jnp.maximum(
            present_classes,
            jnp.asarray(1.0, dtype=predicted_scores.dtype),
        )
        info["is_finite"] = jnp.logical_and(
            jnp.all(jnp.isfinite(predicted_logits)),
            jnp.logical_and(
                jnp.all(jnp.isfinite(encoded_logits)),
                jnp.all(
                    jnp.isfinite(
                        jnp.asarray(tuple(info.values()))
                    )
                ),
            ),
        )
        return info

    @jax.jit
    def evaluate_candidates(self, observations, world_model, agent, rng):
        """Compare critic and predicted-progress candidate rankings.

        This method is read-only. It generates the same best-of-N action
        candidates as the policy, but only returns aggregate diagnostics and
        never changes which action is selected.
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
        progress_logits = self.network(predicted_latents)
        progress_probabilities = jax.nn.softmax(progress_logits, axis=-1)
        progress_values = jnp.arange(
            self.num_classes, dtype=progress_logits.dtype
        )
        progress_scores = jnp.sum(
            progress_probabilities * progress_values, axis=-1
        )

        current_latents = world_model.network(
            observations, method="encode"
        )
        current_latents = jax.lax.stop_gradient(current_latents)
        current_probabilities = jax.nn.softmax(
            self.network(current_latents), axis=-1
        )
        current_scores = jnp.sum(
            current_probabilities * progress_values, axis=-1
        )

        critic_ranks = jnp.argsort(
            jnp.argsort(critic_scores, axis=-1), axis=-1
        ).astype(progress_scores.dtype)
        progress_ranks = jnp.argsort(
            jnp.argsort(progress_scores, axis=-1), axis=-1
        ).astype(progress_scores.dtype)

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
        progress_top1 = jnp.argmax(progress_scores, axis=-1)
        topk = max(1, num_samples // 10)
        _, critic_topk = jax.lax.top_k(critic_scores, topk)
        _, progress_topk = jax.lax.top_k(progress_scores, topk)
        topk_overlap = jnp.mean(
            jax.vmap(
                lambda left, right: jnp.mean(
                    jnp.any(
                        left[:, None] == right[None, :], axis=-1
                    ).astype(progress_scores.dtype)
                )
            )(critic_topk, progress_topk)
        )

        critic_choice_progress_rank = jnp.take_along_axis(
            progress_ranks, critic_top1[..., None], axis=-1
        ).squeeze(-1)
        progress_choice_critic_rank = jnp.take_along_axis(
            critic_ranks, progress_top1[..., None], axis=-1
        ).squeeze(-1)
        rank_denominator = jnp.asarray(
            max(1, num_samples - 1), dtype=progress_scores.dtype
        )
        critic_actions = jnp.take_along_axis(
            candidate_actions,
            critic_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        progress_actions = jnp.take_along_axis(
            candidate_actions,
            progress_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        score_deltas = progress_scores - current_scores[..., None]
        probability_entropy = -jnp.sum(
            progress_probabilities
            * jnp.log(progress_probabilities.clip(1e-8)),
            axis=-1,
        )

        return {
            "spearman_correlation": mean_correlation(
                critic_ranks, progress_ranks
            ),
            "score_correlation": mean_correlation(
                critic_scores, progress_scores
            ),
            "top1_agreement": jnp.mean(
                (critic_top1 == progress_top1).astype(
                    progress_scores.dtype
                )
            ),
            "topk_overlap": topk_overlap,
            "random_top1_agreement": jnp.asarray(
                1.0 / num_samples, dtype=progress_scores.dtype
            ),
            "random_topk_overlap": jnp.asarray(
                topk / num_samples, dtype=progress_scores.dtype
            ),
            "critic_choice_progress_percentile": jnp.mean(
                critic_choice_progress_rank / rank_denominator
            ),
            "progress_choice_critic_percentile": jnp.mean(
                progress_choice_critic_rank / rank_denominator
            ),
            "selected_action_l2": jnp.mean(
                jnp.linalg.norm(
                    critic_actions - progress_actions, axis=-1
                )
            ),
            "critic_score_std": jnp.mean(
                jnp.std(critic_scores, axis=-1)
            ),
            "progress_score_std": jnp.mean(
                jnp.std(progress_scores, axis=-1)
            ),
            "current_progress_mean": jnp.mean(current_scores),
            "candidate_progress_mean": jnp.mean(progress_scores),
            "candidate_progress_delta_mean": jnp.mean(score_deltas),
            "candidate_progress_delta_std": jnp.std(score_deltas),
            "candidate_progress_max_delta_mean": jnp.mean(
                jnp.max(score_deltas, axis=-1)
            ),
            "candidate_probability_entropy": jnp.mean(
                probability_entropy
            ),
            "candidate_max_probability": jnp.mean(
                jnp.max(progress_probabilities, axis=-1)
            ),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(critic_scores)),
                jnp.logical_and(
                    jnp.all(jnp.isfinite(progress_scores)),
                    jnp.all(jnp.isfinite(progress_probabilities)),
                ),
            ),
        }


class LatentPotentialTrainState(flax.struct.PyTreeNode):
    """Independent diagnostic regressor for a continuous task potential."""

    network: Any
    coef: float = nonpytree_field()
    target_mean: float = nonpytree_field()
    target_std: float = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        latent_dim=64,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=1.0,
        target_mean=0.0,
        target_std=1.0,
    ):
        if target_std <= 0:
            raise ValueError("target_std must be positive")
        model = LatentPotentialHead(hidden_dims=hidden_dims)
        params = model.init(
            jax.random.PRNGKey(seed),
            jnp.zeros((latent_dim,), dtype=jnp.float32),
        )["params"]
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adam(learning_rate=learning_rate),
        )
        return cls(
            network=network,
            coef=coef,
            target_mean=float(target_mean),
            target_std=float(target_std),
        )

    def potential_loss(self, batch, world_model, params):
        latents = world_model.network(
            batch["observations"], method="encode"
        )
        latents = jax.lax.stop_gradient(latents)
        normalized_predictions = self.network(latents, params=params)
        targets = batch["wm_potentials"]
        if targets.ndim > normalized_predictions.ndim:
            targets = targets[..., 0]
        targets = targets.astype(normalized_predictions.dtype)
        target_mean = jnp.asarray(
            self.target_mean, dtype=normalized_predictions.dtype
        )
        target_std = jnp.asarray(
            self.target_std, dtype=normalized_predictions.dtype
        )
        normalized_targets = (targets - target_mean) / target_std
        predictions = normalized_predictions * target_std + target_mean

        if "valid" in batch:
            valid = batch["valid"]
            if valid.ndim > predictions.ndim:
                valid = valid[..., 0]
            valid = valid.astype(predictions.dtype)
        else:
            valid = jnp.ones(predictions.shape, dtype=predictions.dtype)
        valid_count = jnp.maximum(
            jnp.sum(valid), jnp.asarray(1.0, dtype=valid.dtype)
        )
        normalized_errors = normalized_predictions - normalized_targets
        loss = jnp.sum(jnp.square(normalized_errors) * valid) / valid_count
        errors = predictions - targets
        mae = jnp.sum(jnp.abs(errors) * valid) / valid_count

        prediction_mean = jnp.sum(predictions * valid) / valid_count
        observed_target_mean = jnp.sum(targets * valid) / valid_count
        centered_predictions = predictions - prediction_mean
        centered_targets = targets - observed_target_mean
        prediction_variance = jnp.sum(
            jnp.square(centered_predictions) * valid
        ) / valid_count
        target_variance = jnp.sum(
            jnp.square(centered_targets) * valid
        ) / valid_count
        covariance = jnp.sum(
            centered_predictions * centered_targets * valid
        ) / valid_count
        correlation = covariance / jnp.sqrt(
            prediction_variance * target_variance
        ).clip(1e-8)
        raw_mse = jnp.sum(jnp.square(errors) * valid) / valid_count
        r2 = 1.0 - raw_mse / target_variance.clip(1e-8)

        info = {
            "loss": loss,
            "mae": mae,
            "correlation": correlation,
            "r2": r2,
            "prediction_mean": prediction_mean,
            "prediction_std": jnp.sqrt(
                jnp.maximum(prediction_variance, 0.0)
            ),
            "target_mean": observed_target_mean,
            "target_std": jnp.sqrt(jnp.maximum(target_variance, 0.0)),
            "valid_fraction": valid.mean(),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(predictions)),
                jnp.all(
                    jnp.isfinite(
                        jnp.asarray((loss, mae, correlation, r2))
                    )
                ),
            ),
        }
        return loss, info

    @staticmethod
    def _update(state, batch, world_model):
        def loss_fn(params):
            loss, info = state.potential_loss(
                batch, world_model, params
            )
            weighted_loss = state.coef * loss
            info["weighted_loss"] = weighted_loss
            return weighted_loss, info

        network, info = state.network.apply_loss_fn(loss_fn)
        return state.replace(network=network), info

    @jax.jit
    def update(self, batch, world_model):
        """Apply one potential-head optimizer update."""
        return self._update(self, batch, world_model)

    @jax.jit
    def batch_update(self, batch, world_model):
        """Apply one potential-head update per leading UTD entry."""

        def scan_update(state, scan_batch):
            return self._update(state, scan_batch, world_model)

        state, infos = jax.lax.scan(scan_update, self, batch)
        infos = jax.tree_util.tree_map(lambda value: value.mean(), infos)
        return state, infos

    @jax.jit
    def evaluate(self, batch, world_model):
        """Evaluate continuous-potential metrics without changing state."""
        _, info = self.potential_loss(
            batch, world_model, self.network.params
        )
        return info

    def _raw_scores(self, latents):
        normalized_scores = self.network(latents)
        return (
            normalized_scores * self.target_std + self.target_mean
        )

    @jax.jit
    def evaluate_predicted_future(self, batch, world_model):
        """Evaluate potential after composing dynamics with the head."""
        predicted_latents, target_latents = world_model.network(
            batch["observations"],
            batch["actions"],
            batch["target_observations"],
        )
        predicted_latents = jax.lax.stop_gradient(predicted_latents)
        target_latents = jax.lax.stop_gradient(target_latents)
        predictions = self._raw_scores(predicted_latents)
        encoded_scores = self._raw_scores(target_latents)
        current_latents = jax.lax.stop_gradient(
            world_model.network(batch["observations"], method="encode")
        )
        current_scores = self._raw_scores(current_latents)
        targets = batch["target_potentials"].astype(predictions.dtype)
        current_targets = batch["current_potentials"].astype(
            predictions.dtype
        )

        def correlation(left, right):
            left = left - left.mean()
            right = right - right.mean()
            return jnp.sum(left * right) / jnp.sqrt(
                jnp.sum(jnp.square(left)) * jnp.sum(jnp.square(right))
            ).clip(1e-8)

        errors = predictions - targets
        target_variance = jnp.var(targets)
        predicted_deltas = predictions - current_scores
        encoded_deltas = encoded_scores - current_scores
        target_deltas = targets - current_targets
        delta_mask = (jnp.abs(target_deltas) > 0.01).astype(
            predictions.dtype
        )
        delta_count = jnp.maximum(
            jnp.sum(delta_mask),
            jnp.asarray(1.0, dtype=predictions.dtype),
        )
        info = {
            "mae": jnp.mean(jnp.abs(errors)),
            "correlation": correlation(predictions, targets),
            "r2": 1.0 - jnp.mean(jnp.square(errors))
            / target_variance.clip(1e-8),
            "prediction_mean": jnp.mean(predictions),
            "prediction_std": jnp.std(predictions),
            "target_mean": jnp.mean(targets),
            "target_std": jnp.std(targets),
            "encoded_mean": jnp.mean(encoded_scores),
            "encoded_std": jnp.std(encoded_scores),
            "score_vs_encoded_mae": jnp.mean(
                jnp.abs(predictions - encoded_scores)
            ),
            "score_vs_encoded_correlation": correlation(
                predictions, encoded_scores
            ),
            "delta_correlation": correlation(
                predicted_deltas, target_deltas
            ),
            "delta_mae": jnp.mean(
                jnp.abs(predicted_deltas - target_deltas)
            ),
            "delta_prediction_mean": jnp.mean(predicted_deltas),
            "delta_prediction_std": jnp.std(predicted_deltas),
            "delta_target_mean": jnp.mean(target_deltas),
            "delta_target_std": jnp.std(target_deltas),
            "delta_vs_encoded_correlation": correlation(
                predicted_deltas, encoded_deltas
            ),
            "encoded_delta_correlation": correlation(
                encoded_deltas, target_deltas
            ),
            "nontrivial_delta_fraction": jnp.mean(delta_mask),
            "nontrivial_delta_sign_agreement": jnp.sum(
                (
                    jnp.sign(predicted_deltas)
                    == jnp.sign(target_deltas)
                ).astype(predictions.dtype)
                * delta_mask
            ) / delta_count,
            "latent_mse": jnp.mean(
                jnp.square(predicted_latents - target_latents)
            ),
            "latent_cosine": jnp.mean(
                jnp.sum(predicted_latents * target_latents, axis=-1)
                / (
                    jnp.linalg.norm(predicted_latents, axis=-1)
                    * jnp.linalg.norm(target_latents, axis=-1)
                ).clip(1e-8)
            ),
        }
        info["is_finite"] = jnp.logical_and(
            jnp.all(jnp.isfinite(predictions)),
            jnp.logical_and(
                jnp.all(jnp.isfinite(encoded_scores)),
                jnp.all(jnp.isfinite(jnp.asarray(tuple(info.values())))),
            ),
        )
        return info

    @jax.jit
    def evaluate_candidates(self, observations, world_model, agent, rng):
        """Compare critic and predicted-potential candidate rankings."""
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
        candidate_actions = jnp.clip(
            agent.compute_flow_actions(candidate_observations, noises),
            -1,
            1,
        )
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
        potential_scores = self._raw_scores(predicted_latents)
        current_latents = jax.lax.stop_gradient(
            world_model.network(observations, method="encode")
        )
        current_scores = self._raw_scores(current_latents)

        critic_ranks = jnp.argsort(
            jnp.argsort(critic_scores, axis=-1), axis=-1
        ).astype(potential_scores.dtype)
        potential_ranks = jnp.argsort(
            jnp.argsort(potential_scores, axis=-1), axis=-1
        ).astype(potential_scores.dtype)

        def mean_correlation(left, right):
            left = left - left.mean(axis=-1, keepdims=True)
            right = right - right.mean(axis=-1, keepdims=True)
            return jnp.mean(
                jnp.sum(left * right, axis=-1)
                / jnp.sqrt(
                    jnp.sum(jnp.square(left), axis=-1)
                    * jnp.sum(jnp.square(right), axis=-1)
                ).clip(1e-8)
            )

        critic_top1 = jnp.argmax(critic_scores, axis=-1)
        potential_top1 = jnp.argmax(potential_scores, axis=-1)
        topk = max(1, num_samples // 10)
        _, critic_topk = jax.lax.top_k(critic_scores, topk)
        _, potential_topk = jax.lax.top_k(potential_scores, topk)
        topk_overlap = jnp.mean(
            jax.vmap(
                lambda left, right: jnp.mean(
                    jnp.any(
                        left[:, None] == right[None, :], axis=-1
                    ).astype(potential_scores.dtype)
                )
            )(critic_topk, potential_topk)
        )
        critic_choice_potential_rank = jnp.take_along_axis(
            potential_ranks, critic_top1[..., None], axis=-1
        ).squeeze(-1)
        potential_choice_critic_rank = jnp.take_along_axis(
            critic_ranks, potential_top1[..., None], axis=-1
        ).squeeze(-1)
        rank_denominator = jnp.asarray(
            max(1, num_samples - 1), dtype=potential_scores.dtype
        )
        critic_actions = jnp.take_along_axis(
            candidate_actions,
            critic_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        potential_actions = jnp.take_along_axis(
            candidate_actions,
            potential_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        score_deltas = potential_scores - current_scores[..., None]
        return {
            "spearman_correlation": mean_correlation(
                critic_ranks, potential_ranks
            ),
            "score_correlation": mean_correlation(
                critic_scores, potential_scores
            ),
            "top1_agreement": jnp.mean(
                (critic_top1 == potential_top1).astype(
                    potential_scores.dtype
                )
            ),
            "topk_overlap": topk_overlap,
            "random_top1_agreement": jnp.asarray(
                1.0 / num_samples, dtype=potential_scores.dtype
            ),
            "random_topk_overlap": jnp.asarray(
                topk / num_samples, dtype=potential_scores.dtype
            ),
            "critic_choice_potential_percentile": jnp.mean(
                critic_choice_potential_rank / rank_denominator
            ),
            "potential_choice_critic_percentile": jnp.mean(
                potential_choice_critic_rank / rank_denominator
            ),
            "selected_action_l2": jnp.mean(
                jnp.linalg.norm(
                    critic_actions - potential_actions, axis=-1
                )
            ),
            "critic_score_std": jnp.mean(
                jnp.std(critic_scores, axis=-1)
            ),
            "candidate_action_std": jnp.mean(
                jnp.std(candidate_actions, axis=-2)
            ),
            "potential_score_std": jnp.mean(
                jnp.std(potential_scores, axis=-1)
            ),
            "current_potential_mean": jnp.mean(current_scores),
            "candidate_potential_mean": jnp.mean(potential_scores),
            "candidate_potential_delta_mean": jnp.mean(score_deltas),
            "candidate_potential_delta_std": jnp.std(score_deltas),
            "candidate_potential_max_delta_mean": jnp.mean(
                jnp.max(score_deltas, axis=-1)
            ),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(critic_scores)),
                jnp.all(jnp.isfinite(potential_scores)),
            ),
        }


class LatentPotentialDeltaTrainState(flax.struct.PyTreeNode):
    """Independent regressor for short-horizon potential changes."""

    network: Any
    coef: float = nonpytree_field()
    target_mean: float = nonpytree_field()
    target_std: float = nonpytree_field()
    nontrivial_threshold: float = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed,
        latent_dim,
        example_action_chunk,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=1.0,
        target_mean=0.0,
        target_std=1.0,
        nontrivial_threshold=0.01,
    ):
        if target_std <= 0:
            raise ValueError("target_std must be positive")
        if nontrivial_threshold <= 0:
            raise ValueError("nontrivial_threshold must be positive")
        model = LatentPotentialDeltaHead(hidden_dims=hidden_dims)
        zero_latent = jnp.zeros((latent_dim,), dtype=jnp.float32)
        params = model.init(
            jax.random.PRNGKey(seed),
            zero_latent,
            zero_latent,
            jnp.zeros_like(example_action_chunk),
        )["params"]
        network = TrainState.create(
            model_def=model,
            params=params,
            tx=optax.adam(learning_rate=learning_rate),
        )
        return cls(
            network=network,
            coef=coef,
            target_mean=float(target_mean),
            target_std=float(target_std),
            nontrivial_threshold=float(nontrivial_threshold),
        )

    def delta_loss(self, batch, world_model, params):
        latents = world_model.network(
            batch["observations"], method="encode"
        )
        predicted_latents = world_model.network(
            batch["observations"],
            batch["actions"],
            method="predict",
        )
        latents = jax.lax.stop_gradient(latents)
        predicted_latents = jax.lax.stop_gradient(predicted_latents)
        normalized_predictions = self.network(
            latents,
            predicted_latents,
            batch["actions"],
            params=params,
        )
        targets = batch["wm_potential_deltas"].astype(
            normalized_predictions.dtype
        )
        target_mean = jnp.asarray(
            self.target_mean, dtype=normalized_predictions.dtype
        )
        target_std = jnp.asarray(
            self.target_std, dtype=normalized_predictions.dtype
        )
        normalized_targets = (targets - target_mean) / target_std
        predictions = normalized_predictions * target_std + target_mean
        normalized_errors = normalized_predictions - normalized_targets
        loss = jnp.mean(jnp.square(normalized_errors))
        errors = predictions - targets
        mae = jnp.mean(jnp.abs(errors))

        centered_predictions = predictions - predictions.mean()
        centered_targets = targets - targets.mean()
        prediction_variance = jnp.mean(jnp.square(centered_predictions))
        target_variance = jnp.mean(jnp.square(centered_targets))
        covariance = jnp.mean(centered_predictions * centered_targets)
        correlation = covariance / jnp.sqrt(
            prediction_variance * target_variance
        ).clip(1e-8)
        raw_mse = jnp.mean(jnp.square(errors))
        r2 = 1.0 - raw_mse / target_variance.clip(1e-8)
        threshold = jnp.asarray(
            self.nontrivial_threshold, dtype=predictions.dtype
        )
        nontrivial = (jnp.abs(targets) > threshold).astype(
            predictions.dtype
        )
        nontrivial_count = jnp.maximum(
            jnp.sum(nontrivial),
            jnp.asarray(1.0, dtype=predictions.dtype),
        )
        sign_agreement = jnp.sum(
            (jnp.sign(predictions) == jnp.sign(targets)).astype(
                predictions.dtype
            )
            * nontrivial
        ) / nontrivial_count

        info = {
            "loss": loss,
            "mae": mae,
            "correlation": correlation,
            "r2": r2,
            "prediction_mean": predictions.mean(),
            "prediction_std": jnp.sqrt(
                jnp.maximum(prediction_variance, 0.0)
            ),
            "target_mean": targets.mean(),
            "target_std": jnp.sqrt(jnp.maximum(target_variance, 0.0)),
            "nontrivial_fraction": nontrivial.mean(),
            "nontrivial_sign_agreement": sign_agreement,
            "decrease_fraction": jnp.mean((targets < -threshold).astype(
                predictions.dtype
            )),
            "neutral_fraction": jnp.mean(
                (jnp.abs(targets) <= threshold).astype(predictions.dtype)
            ),
            "increase_fraction": jnp.mean((targets > threshold).astype(
                predictions.dtype
            )),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(predictions)),
                jnp.all(
                    jnp.isfinite(
                        jnp.asarray((loss, mae, correlation, r2))
                    )
                ),
            ),
        }
        return loss, info

    @staticmethod
    def _update(state, batch, world_model):
        def loss_fn(params):
            loss, info = state.delta_loss(batch, world_model, params)
            weighted_loss = state.coef * loss
            info["weighted_loss"] = weighted_loss
            return weighted_loss, info

        network, info = state.network.apply_loss_fn(loss_fn)
        return state.replace(network=network), info

    @jax.jit
    def update(self, batch, world_model):
        return self._update(self, batch, world_model)

    @jax.jit
    def batch_update(self, batch, world_model):
        def scan_update(state, scan_batch):
            return self._update(state, scan_batch, world_model)

        state, infos = jax.lax.scan(scan_update, self, batch)
        infos = jax.tree_util.tree_map(lambda value: value.mean(), infos)
        return state, infos

    @jax.jit
    def evaluate(self, batch, world_model):
        _, info = self.delta_loss(batch, world_model, self.network.params)
        return info

    @jax.jit
    def evaluate_candidates(self, observations, world_model, agent, rng):
        """Compare critic and predicted-delta rankings read-only."""
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
        candidate_actions = jnp.clip(
            agent.compute_flow_actions(candidate_observations, noises),
            -1,
            1,
        )
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
        latents = world_model.network(
            candidate_observations, method="encode"
        )
        predicted_latents = world_model.network(
            candidate_observations,
            action_chunks,
            method="predict",
        )
        latents = jax.lax.stop_gradient(latents)
        predicted_latents = jax.lax.stop_gradient(predicted_latents)
        normalized_delta_scores = self.network(
            latents, predicted_latents, action_chunks
        )
        delta_scores = (
            normalized_delta_scores * self.target_std + self.target_mean
        )
        critic_ranks = jnp.argsort(
            jnp.argsort(critic_scores, axis=-1), axis=-1
        ).astype(delta_scores.dtype)
        delta_ranks = jnp.argsort(
            jnp.argsort(delta_scores, axis=-1), axis=-1
        ).astype(delta_scores.dtype)

        def mean_correlation(left, right):
            left = left - left.mean(axis=-1, keepdims=True)
            right = right - right.mean(axis=-1, keepdims=True)
            return jnp.mean(
                jnp.sum(left * right, axis=-1)
                / jnp.sqrt(
                    jnp.sum(jnp.square(left), axis=-1)
                    * jnp.sum(jnp.square(right), axis=-1)
                ).clip(1e-8)
            )

        critic_top1 = jnp.argmax(critic_scores, axis=-1)
        delta_top1 = jnp.argmax(delta_scores, axis=-1)
        topk = max(1, num_samples // 10)
        _, critic_topk = jax.lax.top_k(critic_scores, topk)
        _, delta_topk = jax.lax.top_k(delta_scores, topk)
        topk_overlap = jnp.mean(
            jax.vmap(
                lambda left, right: jnp.mean(
                    jnp.any(
                        left[:, None] == right[None, :], axis=-1
                    ).astype(delta_scores.dtype)
                )
            )(critic_topk, delta_topk)
        )
        critic_choice_delta_rank = jnp.take_along_axis(
            delta_ranks, critic_top1[..., None], axis=-1
        ).squeeze(-1)
        delta_choice_critic_rank = jnp.take_along_axis(
            critic_ranks, delta_top1[..., None], axis=-1
        ).squeeze(-1)
        rank_denominator = jnp.asarray(
            max(1, num_samples - 1), dtype=delta_scores.dtype
        )
        critic_actions = jnp.take_along_axis(
            candidate_actions,
            critic_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        delta_actions = jnp.take_along_axis(
            candidate_actions,
            delta_top1[..., None, None],
            axis=-2,
        ).squeeze(-2)
        threshold = jnp.asarray(
            self.nontrivial_threshold, dtype=delta_scores.dtype
        )
        info = {
            "spearman_correlation": mean_correlation(
                critic_ranks, delta_ranks
            ),
            "score_correlation": mean_correlation(
                critic_scores, delta_scores
            ),
            "top1_agreement": jnp.mean(
                (critic_top1 == delta_top1).astype(delta_scores.dtype)
            ),
            "topk_overlap": topk_overlap,
            "random_top1_agreement": jnp.asarray(
                1.0 / num_samples, dtype=delta_scores.dtype
            ),
            "random_topk_overlap": jnp.asarray(
                topk / num_samples, dtype=delta_scores.dtype
            ),
            "critic_choice_delta_percentile": jnp.mean(
                critic_choice_delta_rank / rank_denominator
            ),
            "delta_choice_critic_percentile": jnp.mean(
                delta_choice_critic_rank / rank_denominator
            ),
            "selected_action_l2": jnp.mean(
                jnp.linalg.norm(critic_actions - delta_actions, axis=-1)
            ),
            "critic_score_std": jnp.mean(
                jnp.std(critic_scores, axis=-1)
            ),
            "candidate_action_std": jnp.mean(
                jnp.std(candidate_actions, axis=-2)
            ),
            "delta_score_mean": jnp.mean(delta_scores),
            "delta_score_std": jnp.mean(
                jnp.std(delta_scores, axis=-1)
            ),
            "delta_score_global_std": jnp.std(delta_scores),
            "max_delta_mean": jnp.mean(
                jnp.max(delta_scores, axis=-1)
            ),
            "positive_candidate_fraction": jnp.mean(
                (delta_scores > threshold).astype(delta_scores.dtype)
            ),
            "states_with_positive_candidate": jnp.mean(
                jnp.any(delta_scores > threshold, axis=-1).astype(
                    delta_scores.dtype
                )
            ),
            "is_finite": jnp.logical_and(
                jnp.all(jnp.isfinite(critic_scores)),
                jnp.all(jnp.isfinite(delta_scores)),
            ),
        }
        for label, veto_threshold in (
            ("m002", -0.02),
            ("m001", -0.01),
            ("zero", 0.0),
        ):
            safe = delta_scores >= veto_threshold
            safe_count = jnp.sum(safe, axis=-1)
            enough_safe = safe_count >= 1
            safe_top1 = jnp.argmax(
                jnp.where(safe, critic_scores, -jnp.inf), axis=-1
            )
            chosen = jnp.where(enough_safe, safe_top1, critic_top1)
            chosen_delta = jnp.take_along_axis(
                delta_scores, chosen[..., None], axis=-1
            ).squeeze(-1)
            baseline_delta = jnp.take_along_axis(
                delta_scores, critic_top1[..., None], axis=-1
            ).squeeze(-1)
            chosen_critic_rank = jnp.take_along_axis(
                critic_ranks, chosen[..., None], axis=-1
            ).squeeze(-1)
            prefix = f"wm_delta_veto/{label}"
            info.update(
                {
                    f"{prefix}/threshold": jnp.asarray(
                        veto_threshold, dtype=delta_scores.dtype
                    ),
                    f"{prefix}/rejected_fraction": jnp.mean(
                        (~safe).astype(delta_scores.dtype)
                    ),
                    f"{prefix}/states_with_rejection": jnp.mean(
                        jnp.any(~safe, axis=-1).astype(delta_scores.dtype)
                    ),
                    f"{prefix}/all_rejected_fraction": jnp.mean(
                        (safe_count == 0).astype(delta_scores.dtype)
                    ),
                    f"{prefix}/chosen_delta": jnp.mean(chosen_delta),
                    f"{prefix}/baseline_chosen_delta": jnp.mean(
                        baseline_delta
                    ),
                    f"{prefix}/baseline_rejected_fraction": jnp.mean(
                        (baseline_delta < veto_threshold).astype(
                            delta_scores.dtype
                        )
                    ),
                    f"{prefix}/action_changed_fraction": jnp.mean(
                        (chosen != critic_top1).astype(delta_scores.dtype)
                    ),
                    f"{prefix}/chosen_critic_percentile": jnp.mean(
                        chosen_critic_rank / rank_denominator
                    ),
                }
            )
        return info

    @jax.jit
    def sample_actions(
        self,
        observations,
        world_model,
        agent,
        rng,
        score_lambda=0.0,
    ):
        """Select best-of-N actions using critic plus predicted delta.

        A zero coefficient follows the original critic-only argmax exactly.
        This read-only method is separate from `ACFQLAgent.sample_actions`, so
        baseline training, online collection, and default evaluation remain
        unchanged.
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
        candidate_actions = jnp.clip(
            agent.compute_flow_actions(candidate_observations, noises),
            -1,
            1,
        )
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
        latents = world_model.network(
            candidate_observations, method="encode"
        )
        predicted_latents = world_model.network(
            candidate_observations,
            action_chunks,
            method="predict",
        )
        normalized_delta_scores = self.network(
            latents, predicted_latents, action_chunks
        )
        delta_scores = (
            normalized_delta_scores * self.target_std + self.target_mean
        )
        normalized_critic_scores = (
            critic_scores - critic_scores.mean(axis=-1, keepdims=True)
        ) / jnp.std(critic_scores, axis=-1, keepdims=True).clip(1e-6)
        normalized_delta_scores = (
            delta_scores - delta_scores.mean(axis=-1, keepdims=True)
        ) / jnp.std(delta_scores, axis=-1, keepdims=True).clip(1e-6)
        mixed_scores = (
            normalized_critic_scores
            + score_lambda * normalized_delta_scores
        )
        selection_scores = jax.lax.cond(
            jnp.asarray(score_lambda) == 0,
            lambda: critic_scores,
            lambda: mixed_scores,
        )
        indices = jnp.argmax(selection_scores, axis=-1)

        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        batch_size = len(flat_indices)
        return candidate_actions.reshape(
            (-1, num_samples, flat_action_dim)
        )[jnp.arange(batch_size), flat_indices, :].reshape(
            batch_shape + (flat_action_dim,)
        )


    @jax.jit
    def sample_actions_with_veto(
        self,
        observations,
        world_model,
        agent,
        rng,
        threshold=-0.01,
        min_candidates=1,
    ):
        """Keep critic ranking while filtering predicted negative deltas."""
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
        candidate_actions = jnp.clip(
            agent.compute_flow_actions(candidate_observations, noises),
            -1,
            1,
        )
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
        latents = world_model.network(
            candidate_observations, method="encode"
        )
        predicted_latents = world_model.network(
            candidate_observations,
            action_chunks,
            method="predict",
        )
        normalized_delta_scores = self.network(
            latents, predicted_latents, action_chunks
        )
        delta_scores = (
            normalized_delta_scores * self.target_std + self.target_mean
        )
        safe = delta_scores >= threshold
        safe_count = jnp.sum(safe, axis=-1)
        safe_top1 = jnp.argmax(
            jnp.where(safe, critic_scores, -jnp.inf), axis=-1
        )
        critic_top1 = jnp.argmax(critic_scores, axis=-1)
        indices = jnp.where(
            safe_count >= min_candidates, safe_top1, critic_top1
        )

        batch_shape = indices.shape
        flat_indices = indices.reshape(-1)
        batch_size = len(flat_indices)
        return candidate_actions.reshape(
            (-1, num_samples, flat_action_dim)
        )[jnp.arange(batch_size), flat_indices, :].reshape(
            batch_shape + (flat_action_dim,)
        )
