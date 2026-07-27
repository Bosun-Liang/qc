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
