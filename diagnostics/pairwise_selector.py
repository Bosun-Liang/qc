"""Utilities for the evaluation-only Pairwise Selector v1 experiment."""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diagnostics.branch_label_informativeness import percentile, rank_correlation
from utils.flax_utils import TrainState
from utils.networks import MLP


NUM_CANDIDATES = 8
OBSERVATION_DIM = 46
ACTION_CHUNK_SHAPE = (5, 5)
ACTION_DIM = 25
PAIR_INDICES = np.asarray(
    [(left, right) for left in range(NUM_CANDIDATES) for right in range(left + 1, NUM_CANDIDATES)],
    dtype=np.int32,
)
INPUT_DIMS = {
    "full": OBSERVATION_DIM + ACTION_DIM,
    "observation_only": OBSERVATION_DIM,
    "action_only": ACTION_DIM,
    "shuffle": OBSERVATION_DIM + ACTION_DIM,
}


class PairwiseSelector(nn.Module):
    """Small scalar MLP matching the project's existing GELU MLP style."""

    hidden_dims: tuple[int, ...] = (128, 128)

    @nn.compact
    def __call__(self, inputs):
        return MLP((*self.hidden_dims, 1), activations=nn.gelu)(inputs)[..., 0]


@dataclass(frozen=True)
class GroupedSplit:
    name: str
    state_ids: np.ndarray
    episode_ids: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    critic_scores: np.ndarray
    baseline_episode_success: np.ndarray


@dataclass(frozen=True)
class Normalization:
    observation_mean: np.ndarray
    observation_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    observation_near_zero: np.ndarray
    action_near_zero: np.ndarray


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_dataset(path):
    """Load the frozen NPZ and validate all state/episode grouping invariants."""
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    required = {
        "state_id",
        "episode_id",
        "candidate_index",
        "observation",
        "candidate_action_chunk",
        "critic_score",
        "discounted_return",
        "split",
        "reward_mask",
        "executed_steps",
        "baseline_episode_success",
    }
    missing = required.difference(arrays)
    if missing:
        raise ValueError(f"Dataset is missing fields: {sorted(missing)}")
    rows = len(arrays["state_id"])
    if arrays["observation"].shape != (rows, OBSERVATION_DIM):
        raise ValueError(f"Unexpected observation shape: {arrays['observation'].shape}")
    if arrays["candidate_action_chunk"].shape != (rows, *ACTION_CHUNK_SHAPE):
        raise ValueError(f"Unexpected action shape: {arrays['candidate_action_chunk'].shape}")
    if arrays["reward_mask"].shape != (rows, 85):
        raise ValueError(f"Unexpected reward_mask shape: {arrays['reward_mask'].shape}")
    if not np.array_equal(arrays["reward_mask"].sum(axis=1), arrays["executed_steps"]):
        raise ValueError("reward_mask and executed_steps disagree")
    for key, value in arrays.items():
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN/Inf")

    grouped = {}
    state_to_split = {}
    episode_to_split = {}
    for state_id in np.unique(arrays["state_id"]):
        indices = np.flatnonzero(arrays["state_id"] == state_id)
        if len(indices) != NUM_CANDIDATES:
            raise ValueError(f"state {state_id} has {len(indices)} candidates")
        order = np.argsort(arrays["candidate_index"][indices])
        indices = indices[order]
        if not np.array_equal(arrays["candidate_index"][indices], np.arange(NUM_CANDIDATES)):
            raise ValueError(f"state {state_id} has incomplete candidate_index")
        if not np.array_equal(
            arrays["observation"][indices],
            np.repeat(arrays["observation"][indices[:1]], NUM_CANDIDATES, axis=0),
        ):
            raise ValueError(f"state {state_id} has inconsistent observations")
        state_splits = np.unique(arrays["split"][indices])
        episode_ids = np.unique(arrays["episode_id"][indices])
        if len(state_splits) != 1 or len(episode_ids) != 1:
            raise ValueError(f"state {state_id} crosses split or episode")
        split = str(state_splits[0])
        episode_id = int(episode_ids[0])
        state_to_split[int(state_id)] = split
        previous = episode_to_split.setdefault(episode_id, split)
        if previous != split:
            raise ValueError(f"episode {episode_id} crosses split")
        grouped[int(state_id)] = indices

    result = {}
    for split in ("train", "validation", "test"):
        state_ids = np.asarray(
            sorted(state_id for state_id, value in state_to_split.items() if value == split),
            dtype=np.int64,
        )
        indices = np.stack([grouped[int(state_id)] for state_id in state_ids])
        first = indices[:, 0]
        result[split] = GroupedSplit(
            name=split,
            state_ids=state_ids,
            episode_ids=arrays["episode_id"][first],
            observations=arrays["observation"][first].astype(np.float32),
            actions=arrays["candidate_action_chunk"][indices].reshape(len(state_ids), NUM_CANDIDATES, ACTION_DIM).astype(np.float32),
            returns=arrays["discounted_return"][indices].astype(np.float32),
            critic_scores=arrays["critic_score"][indices].astype(np.float32),
            baseline_episode_success=arrays["baseline_episode_success"][first].astype(bool),
        )
    return arrays, result


def compute_normalization(train_split, std_floor=1e-6):
    observation_mean = train_split.observations.mean(axis=0, dtype=np.float64)
    observation_raw_std = train_split.observations.std(axis=0, dtype=np.float64)
    action_values = train_split.actions.reshape(-1, ACTION_DIM)
    action_mean = action_values.mean(axis=0, dtype=np.float64)
    action_raw_std = action_values.std(axis=0, dtype=np.float64)
    observation_near_zero = observation_raw_std < std_floor
    action_near_zero = action_raw_std < std_floor
    return Normalization(
        observation_mean=observation_mean.astype(np.float32),
        observation_std=np.maximum(observation_raw_std, std_floor).astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_std=np.maximum(action_raw_std, std_floor).astype(np.float32),
        observation_near_zero=np.flatnonzero(observation_near_zero).astype(np.int32),
        action_near_zero=np.flatnonzero(action_near_zero).astype(np.int32),
    )


def normalized_inputs(split, normalization, mode):
    observations = (
        split.observations - normalization.observation_mean
    ) / normalization.observation_std
    observations = np.repeat(observations[:, None, :], NUM_CANDIDATES, axis=1)
    actions = (split.actions - normalization.action_mean) / normalization.action_std
    if mode in ("full", "shuffle"):
        inputs = np.concatenate([observations, actions], axis=-1)
    elif mode == "observation_only":
        inputs = observations
    elif mode == "action_only":
        inputs = actions
    else:
        raise ValueError(f"Unknown input mode: {mode}")
    if not np.isfinite(inputs).all():
        raise ValueError(f"Normalized {mode} inputs contain NaN/Inf")
    return inputs.astype(np.float32)


def create_train_state(mode, seed, learning_rate=3e-4, weight_decay=1e-4):
    model = PairwiseSelector()
    input_dim = INPUT_DIMS[mode]
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1, input_dim), dtype=jnp.float32))["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return TrainState.create(model, params, tx=tx)


def count_parameters(params):
    return int(sum(np.prod(value.shape) for value in jax.tree_util.tree_leaves(params)))


def pair_mask_and_sign(returns, epsilon):
    left = jnp.asarray(PAIR_INDICES[:, 0])
    right = jnp.asarray(PAIR_INDICES[:, 1])
    delta = returns[:, left] - returns[:, right]
    absolute_delta = jnp.abs(delta)
    threshold = jnp.where(epsilon == 0, absolute_delta > 0, absolute_delta >= epsilon)
    return threshold, jnp.sign(delta), left, right


def pairwise_objective(params, apply_fn, inputs, returns, epsilon):
    batch_size = inputs.shape[0]
    scores = apply_fn({"params": params}, inputs.reshape(-1, inputs.shape[-1])).reshape(batch_size, NUM_CANDIDATES)
    mask, signs, left, right = pair_mask_and_sign(returns, epsilon)
    logits = (scores[:, left] - scores[:, right]) * signs
    pair_count = jnp.maximum(mask.sum(), 1)
    loss = (jax.nn.softplus(-logits) * mask).sum() / pair_count
    correct = jnp.where(logits > 0, 1.0, jnp.where(logits == 0, 0.5, 0.0))
    accuracy = (correct * mask).sum() / pair_count
    return loss, {
        "loss": loss,
        "pairwise_accuracy": accuracy,
        "pair_count": mask.sum(),
        "score_mean": scores.mean(),
        "score_std": scores.std(),
    }


@jax.jit
def train_step(state, inputs, returns, epsilon):
    def loss_fn(params):
        return pairwise_objective(params, state.apply_fn, inputs, returns, epsilon)

    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    finite = jnp.asarray(
        [jnp.all(jnp.isfinite(value)) for value in jax.tree_util.tree_leaves(grads)]
    ).all()
    grad_norm = optax.global_norm(grads)
    state = state.apply_gradients(grads=grads)
    return state, {**metrics, "grad_norm": grad_norm, "gradient_finite": finite}


@jax.jit
def evaluate_objective(state, inputs, returns, epsilon):
    return pairwise_objective(state.params, state.apply_fn, inputs, returns, epsilon)[1]


@jax.jit
def predict_scores(state, inputs):
    return state(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape[0], NUM_CANDIDATES)


def ranking_metrics(returns, scores):
    pair_correct = 0.0
    pair_count = 0
    correlations = []
    top1_correct = 0
    regrets = []
    non_tied_regrets = []
    selected_percentiles = []
    true_best_score_percentiles = []
    score_spreads = []
    tied_states = 0
    for labels, predictions in zip(returns, scores):
        labels = np.asarray(labels, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        spread = float(np.ptp(labels))
        tied_states += spread == 0
        correlation = rank_correlation(labels, predictions)
        if np.isfinite(correlation):
            correlations.append(correlation)
        selected = int(np.argmax(predictions))
        true_best = labels == np.max(labels)
        top1_correct += bool(true_best[selected])
        regret = float(np.max(labels) - labels[selected])
        regrets.append(regret)
        if spread > 0:
            non_tied_regrets.append(regret)
        selected_percentiles.append(percentile(labels[selected], labels))
        true_best_score_percentiles.append(
            max(percentile(predictions[index], predictions) for index in np.flatnonzero(true_best))
        )
        score_spreads.append(float(np.ptp(predictions)))
        for left, right in PAIR_INDICES:
            label_delta = labels[left] - labels[right]
            if label_delta == 0:
                continue
            score_delta = predictions[left] - predictions[right]
            pair_correct += 0.5 if score_delta == 0 else float(np.sign(score_delta) == np.sign(label_delta))
            pair_count += 1
    state_count = len(returns)
    return {
        "states": state_count,
        "pair_count": pair_count,
        "pairwise_accuracy": float(pair_correct / pair_count),
        "spearman": float(np.mean(correlations)) if correlations else None,
        "spearman_states": len(correlations),
        "top1_accuracy": float(top1_correct / state_count),
        "top1_regret": float(np.mean(regrets)),
        "non_tied_top1_regret": float(np.mean(non_tied_regrets)) if non_tied_regrets else None,
        "selected_true_return_percentile": float(np.mean(selected_percentiles)),
        "true_best_predicted_score_percentile": float(np.mean(true_best_score_percentiles)),
        "tied_states": int(tied_states),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "mean_score_spread": float(np.mean(score_spreads)),
        "median_score_spread": float(np.median(score_spreads)),
    }


def pair_statistics(grouped, epsilons=(0.0, 0.5, 1.0, 2.0)):
    result = {}
    for split, values in grouped.items():
        deltas = np.abs(
            values.returns[:, PAIR_INDICES[:, 0]] - values.returns[:, PAIR_INDICES[:, 1]]
        )
        nonzero = deltas[deltas > 0]
        result[split] = {
            "states": len(values.state_ids),
            "episodes": len(np.unique(values.episode_ids)),
            "tied_states": int(np.sum(np.ptp(values.returns, axis=1) == 0)),
            "epsilon_pair_counts": {
                str(epsilon): int(np.sum(deltas > 0 if epsilon == 0 else deltas >= epsilon))
                for epsilon in epsilons
            },
            "absolute_return_difference": {
                "all_min": float(np.min(deltas)),
                "all_median": float(np.median(deltas)),
                "all_max": float(np.max(deltas)),
                "nonzero_count": int(len(nonzero)),
                "nonzero_mean": float(np.mean(nonzero)) if len(nonzero) else None,
                "nonzero_quantiles": (
                    [float(value) for value in np.quantile(nonzero, [0, 0.25, 0.5, 0.75, 1])]
                    if len(nonzero)
                    else None
                ),
            },
        }
    return result


def save_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = flax.serialization.to_state_dict(state)
    with open(path, "wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)


def restore_state(path, template):
    with open(path, "rb") as file:
        payload = pickle.load(file)
    return flax.serialization.from_state_dict(template, payload)
