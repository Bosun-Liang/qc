"""Smoke test for the action-conditioned latent potential-delta head."""

import jax
import jax.numpy as jnp

from agents.acfql import ACFQLAgent, get_config
from agents.world_model import (
    LatentPotentialDeltaTrainState,
    WorldModelTrainState,
)


def main():
    batch_size = 64
    horizon_length = 5
    observation_dim = 46
    action_dim = 5

    observation_rng, action_rng, next_observation_rng = jax.random.split(
        jax.random.PRNGKey(0), 3
    )
    observations = jax.random.normal(
        observation_rng, (batch_size, observation_dim)
    )
    actions = jax.random.uniform(
        action_rng,
        (batch_size, horizon_length, action_dim),
        minval=-1.0,
        maxval=1.0,
    )
    next_observations = jax.random.normal(
        next_observation_rng,
        (batch_size, horizon_length, observation_dim),
    )
    deltas = 0.04 * jnp.tanh(
        observations[:, 0]
        + 0.2 * jnp.sum(actions[..., 0], axis=-1)
    )
    batch = {
        "observations": observations,
        "actions": actions,
        "wm_potential_deltas": deltas,
    }

    world_model = WorldModelTrainState.create(
        seed=1,
        example_observations=observations[0],
        example_actions=actions[0, 0],
        horizon_length=horizon_length,
        latent_dim=16,
        hidden_dims=(32, 32),
    )
    delta = LatentPotentialDeltaTrainState.create(
        seed=2,
        latent_dim=16,
        example_action_chunk=actions[0],
        hidden_dims=(32, 32),
        target_mean=float(deltas.mean()),
        target_std=float(deltas.std()),
        nontrivial_threshold=0.01,
    )
    config = get_config()
    config.horizon_length = horizon_length
    config.actor_type = "best-of-n"
    config.actor_num_samples = 4
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.flow_steps = 2
    agent = ACFQLAgent.create(
        0, observations[0], actions[0, 0], config
    )

    predicted_only = world_model.network(
        observations, actions, method="predict"
    )
    predicted_with_target, _ = world_model.network(
        observations,
        actions,
        next_observations[:, -1],
    )
    assert bool(jnp.array_equal(predicted_only, predicted_with_target))

    world_model_before = jax.tree_util.tree_map(jnp.copy, world_model)
    initial_info = delta.evaluate(batch, world_model)
    for _ in range(50):
        delta, info = delta.update(batch, world_model)
        assert bool(info["is_finite"])
        assert bool(jnp.isfinite(info["grad/norm"]))

    final_info = delta.evaluate(batch, world_model)
    assert float(final_info["loss"]) < float(initial_info["loss"])
    assert -1.0 <= float(final_info["correlation"]) <= 1.0
    assert 0.0 <= float(
        final_info["nontrivial_sign_agreement"]
    ) <= 1.0
    assert bool(final_info["is_finite"])
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(world_model_before),
            jax.tree_util.tree_leaves(world_model),
        )
    )

    delta_before_candidates = jax.tree_util.tree_map(jnp.copy, delta)
    candidate_info = delta.evaluate_candidates(
        observations,
        world_model,
        agent,
        jax.random.PRNGKey(3),
    )
    assert bool(candidate_info["is_finite"])
    assert -1.0 <= float(candidate_info["spearman_correlation"]) <= 1.0
    assert -1.0 <= float(candidate_info["score_correlation"]) <= 1.0
    assert 0.0 <= float(candidate_info["top1_agreement"]) <= 1.0
    assert 0.0 <= float(candidate_info["topk_overlap"]) <= 1.0
    assert 0.0 <= float(
        candidate_info["critic_choice_delta_percentile"]
    ) <= 1.0
    assert 0.0 <= float(
        candidate_info["delta_choice_critic_percentile"]
    ) <= 1.0
    assert 0.0 <= float(
        candidate_info["positive_candidate_fraction"]
    ) <= 1.0
    assert 0.0 <= float(
        candidate_info["states_with_positive_candidate"]
    ) <= 1.0
    assert all(
        bool(jnp.isfinite(value)) for value in candidate_info.values()
    )
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(delta_before_candidates),
            jax.tree_util.tree_leaves(delta),
        )
    )

    selection_rng = jax.random.PRNGKey(4)
    baseline_actions = agent.sample_actions(
        observations=observations, rng=selection_rng
    )
    zero_lambda_actions = delta.sample_actions(
        observations,
        world_model,
        agent,
        selection_rng,
        score_lambda=0.0,
    )
    mixed_actions = delta.sample_actions(
        observations,
        world_model,
        agent,
        selection_rng,
        score_lambda=0.1,
    )
    assert bool(jnp.array_equal(baseline_actions, zero_lambda_actions))
    assert baseline_actions.shape == mixed_actions.shape
    assert bool(jnp.all(jnp.isfinite(mixed_actions)))
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(delta_before_candidates),
            jax.tree_util.tree_leaves(delta),
        )
    )

    utd_batch = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value), axis=0), batch
    )
    step_before = delta.network.step
    delta, batch_info = delta.batch_update(utd_batch, world_model)
    assert delta.network.step == step_before + 2
    assert bool(batch_info["is_finite"])

    print("latent delta smoke test passed")
    print(f"initial_loss: {float(initial_info['loss']):.6f}")
    print(f"final_loss: {float(final_info['loss']):.6f}")
    print(f"correlation: {float(final_info['correlation']):.6f}")
    print(f"r2: {float(final_info['r2']):.6f}")


if __name__ == "__main__":
    main()
