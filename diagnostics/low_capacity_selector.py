"""Low-capacity pairwise selector models and training primitives."""

from dataclasses import replace

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diagnostics.pairwise_selector import (
    ACTION_DIM,
    NUM_CANDIDATES,
    OBSERVATION_DIM,
    PAIR_INDICES,
    GroupedSplit,
)
from utils.flax_utils import TrainState
from utils.networks import MLP


MODEL_NAMES = ("bilinear_r8", "small_mlp", "linear_action")
PARAMETER_COUNTS = {
    "bilinear_r8": 594,
    "small_mlp": 3393,
    "linear_action": 26,
}


class LowRankBilinearSelector(nn.Module):
    """Action-linear score plus a rank-r state-action interaction."""

    rank: int = 8

    @nn.compact
    def __call__(self, observations, actions):
        observation_features = nn.Dense(
            self.rank, use_bias=False, name="observation_projection"
        )(observations)
        action_features = nn.Dense(
            self.rank, use_bias=False, name="action_projection"
        )(actions)
        action_weight = self.param(
            "action_weight", nn.initializers.zeros_init(), (ACTION_DIM,)
        )
        bias = self.param("bias", nn.initializers.zeros_init(), ())
        return (
            jnp.sum(actions * action_weight, axis=-1)
            + jnp.sum(observation_features * action_features, axis=-1)
            + bias
        )


class SmallMLPSelector(nn.Module):
    """A 71 -> 32 -> 32 -> 1 GELU selector."""

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return MLP((32, 32, 1), activations=nn.gelu)(inputs)[..., 0]


class LinearActionSelector(nn.Module):
    """A 25 -> 1 action-only linear baseline."""

    @nn.compact
    def __call__(self, observations, actions):
        del observations
        return nn.Dense(1, name="action_score")(actions)[..., 0]


def model_definition(model_name):
    if model_name == "bilinear_r8":
        return LowRankBilinearSelector(rank=8)
    if model_name == "small_mlp":
        return SmallMLPSelector()
    if model_name == "linear_action":
        return LinearActionSelector()
    raise ValueError(f"Unknown model: {model_name}")


def create_low_capacity_state(
    model_name, seed=0, learning_rate=3e-4, weight_decay=1e-4
):
    model = model_definition(model_name)
    observations = jnp.zeros(
        (1, NUM_CANDIDATES, OBSERVATION_DIM), dtype=jnp.float32
    )
    actions = jnp.zeros((1, NUM_CANDIDATES, ACTION_DIM), dtype=jnp.float32)
    params = model.init(
        jax.random.PRNGKey(seed), observations, actions
    )["params"]
    state = TrainState.create(
        model,
        params,
        tx=optax.adamw(
            learning_rate=learning_rate, weight_decay=weight_decay
        ),
    )
    count = int(
        sum(
            np.prod(value.shape)
            for value in jax.tree_util.tree_leaves(params)
        )
    )
    if count != PARAMETER_COUNTS[model_name]:
        raise AssertionError(
            f"{model_name} parameter count {count} != expected "
            f"{PARAMETER_COUNTS[model_name]}"
        )
    return state


def normalized_state_inputs(split, normalization):
    observations = (
        split.observations - normalization.observation_mean
    ) / normalization.observation_std
    observations = np.repeat(
        observations[:, None, :], NUM_CANDIDATES, axis=1
    ).astype(np.float32)
    actions = (
        split.actions - normalization.action_mean
    ) / normalization.action_std
    actions = actions.astype(np.float32)
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("Normalized low-capacity inputs contain NaN/Inf")
    return observations, actions


def subset_split(split, state_indices, name):
    state_indices = np.asarray(state_indices, dtype=np.int64)
    return replace(
        split,
        name=name,
        state_ids=split.state_ids[state_indices],
        episode_ids=split.episode_ids[state_indices],
        observations=split.observations[state_indices],
        actions=split.actions[state_indices],
        returns=split.returns[state_indices],
        critic_scores=split.critic_scores[state_indices],
        baseline_episode_success=split.baseline_episode_success[state_indices],
    )


def build_nested_episode_subsets(train_split, seed=0):
    """Build 30/50/100/140-state nested subsets without splitting episodes."""
    episodes = np.unique(train_split.episode_ids)
    failure_episodes = [
        int(episode)
        for episode in episodes
        if not np.all(
            train_split.baseline_episode_success[
                train_split.episode_ids == episode
            ]
        )
    ]
    success_episodes = [
        int(episode) for episode in episodes if int(episode) not in failure_episodes
    ]
    rng = np.random.default_rng(seed + 404)
    ordered = failure_episodes + list(rng.permutation(success_episodes))
    episode_counts = {25: 3, 50: 5, 100: 10, 140: len(episodes)}
    subsets = {}
    metadata = {}
    previous = set()
    for target, count in episode_counts.items():
        selected_episodes = ordered[:count]
        indices = np.flatnonzero(np.isin(train_split.episode_ids, selected_episodes))
        state_ids = set(int(value) for value in train_split.state_ids[indices])
        if not previous.issubset(state_ids):
            raise AssertionError("Learning-curve subsets are not nested")
        previous = state_ids
        subset = subset_split(train_split, indices, f"train_target_{target}")
        subsets[target] = subset
        metadata[target] = {
            "target_states": target,
            "actual_states": len(indices),
            "episode_ids": [int(value) for value in selected_episodes],
            "episodes": len(selected_episodes),
        }
    if set(subsets[140].state_ids) != set(train_split.state_ids):
        raise AssertionError("n140 subset does not equal the full train split")
    return subsets, metadata


def pairwise_objective(params, apply_fn, observations, actions, returns, epsilon):
    scores = apply_fn({"params": params}, observations, actions)
    left = jnp.asarray(PAIR_INDICES[:, 0])
    right = jnp.asarray(PAIR_INDICES[:, 1])
    return_delta = returns[:, left] - returns[:, right]
    absolute_delta = jnp.abs(return_delta)
    mask = jnp.where(
        epsilon == 0, absolute_delta > 0, absolute_delta >= epsilon
    )
    signs = jnp.sign(return_delta)
    logits = (scores[:, left] - scores[:, right]) * signs
    pair_count = jnp.maximum(mask.sum(), 1)
    loss = (jax.nn.softplus(-logits) * mask).sum() / pair_count
    correct = jnp.where(
        logits > 0, 1.0, jnp.where(logits == 0, 0.5, 0.0)
    )
    return loss, {
        "loss": loss,
        "pairwise_accuracy": (correct * mask).sum() / pair_count,
        "pair_count": mask.sum(),
        "score_mean": scores.mean(),
        "score_std": scores.std(),
    }


@jax.jit
def low_capacity_train_step(
    state, observations, actions, returns, epsilon
):
    def loss_fn(params):
        return pairwise_objective(
            params,
            state.apply_fn,
            observations,
            actions,
            returns,
            epsilon,
        )

    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    gradient_finite = jnp.asarray(
        [
            jnp.all(jnp.isfinite(value))
            for value in jax.tree_util.tree_leaves(grads)
        ]
    ).all()
    state = state.apply_gradients(grads=grads)
    return state, {
        **metrics,
        "grad_norm": optax.global_norm(grads),
        "gradient_finite": gradient_finite,
    }


@jax.jit
def low_capacity_evaluate_objective(
    state, observations, actions, returns, epsilon
):
    return pairwise_objective(
        state.params,
        state.apply_fn,
        observations,
        actions,
        returns,
        epsilon,
    )[1]


@jax.jit
def low_capacity_predict(state, observations, actions):
    return state(observations, actions)


def margin_bucket_accuracy(returns, scores):
    buckets = {
        "[1,2)": (1.0, 2.0),
        "[2,4)": (2.0, 4.0),
        "[4,inf)": (4.0, np.inf),
    }
    result = {}
    for name, (lower, upper) in buckets.items():
        correct = 0.0
        count = 0
        for labels, predictions in zip(returns, scores):
            for left, right in PAIR_INDICES:
                label_delta = float(labels[left] - labels[right])
                magnitude = abs(label_delta)
                if magnitude < lower or magnitude >= upper:
                    continue
                score_delta = float(predictions[left] - predictions[right])
                correct += (
                    0.5
                    if score_delta == 0
                    else float(np.sign(score_delta) == np.sign(label_delta))
                )
                count += 1
        result[name] = {
            "pair_count": count,
            "pairwise_accuracy": float(correct / count) if count else None,
        }
    return result
