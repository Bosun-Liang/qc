"""Smoke test for the continuous latent task-potential head."""

import jax
import jax.numpy as jnp

from agents.acfql import ACFQLAgent, get_config
from agents.world_model import LatentPotentialTrainState, WorldModelTrainState


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
    potentials = 3.0 * jax.nn.sigmoid(
        observations[:, 0] + 0.5 * observations[:, 1]
    )
    batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": jnp.zeros((batch_size, horizon_length)),
        "wm_potentials": jnp.repeat(
            potentials[:, None], horizon_length, axis=1
        ),
        "valid": jnp.ones((batch_size, horizon_length)),
    }

    world_model = WorldModelTrainState.create(
        seed=1,
        example_observations=observations[0],
        example_actions=actions[0, 0],
        horizon_length=horizon_length,
        latent_dim=16,
        hidden_dims=(32, 32),
    )
    potential = LatentPotentialTrainState.create(
        seed=2,
        latent_dim=16,
        hidden_dims=(32, 32),
        target_mean=float(potentials.mean()),
        target_std=float(potentials.std()),
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
    world_model_before = jax.tree_util.tree_map(jnp.copy, world_model)
    initial_info = potential.evaluate(batch, world_model)

    for _ in range(50):
        potential, info = potential.update(batch, world_model)
        assert bool(info["is_finite"])
        assert bool(jnp.isfinite(info["grad/norm"]))

    final_info = potential.evaluate(batch, world_model)
    assert float(final_info["loss"]) < float(initial_info["loss"])
    assert float(final_info["mae"]) >= 0.0
    assert -1.0 <= float(final_info["correlation"]) <= 1.0
    assert bool(final_info["is_finite"])
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(world_model_before),
            jax.tree_util.tree_leaves(world_model),
        )
    )

    transition_batch = {
        "observations": observations,
        "wm_potentials": potentials,
    }
    transition_info = potential.evaluate(transition_batch, world_model)
    assert bool(transition_info["is_finite"])

    potential_before_diagnostics = jax.tree_util.tree_map(
        jnp.copy, potential
    )
    future_info = potential.evaluate_predicted_future(
        {
            "observations": observations,
            "actions": actions,
            "target_observations": next_observations[:, -1],
            "target_potentials": jnp.roll(potentials, 1),
            "current_potentials": potentials,
        },
        world_model,
    )
    assert bool(future_info["is_finite"])
    assert -1.0 <= float(future_info["correlation"]) <= 1.0
    assert -1.0 <= float(
        future_info["score_vs_encoded_correlation"]
    ) <= 1.0
    assert -1.0 <= float(future_info["latent_cosine"]) <= 1.0

    candidate_info = potential.evaluate_candidates(
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
        candidate_info["critic_choice_potential_percentile"]
    ) <= 1.0
    assert 0.0 <= float(
        candidate_info["potential_choice_critic_percentile"]
    ) <= 1.0
    assert all(
        bool(jnp.isfinite(value)) for value in candidate_info.values()
    )
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(potential_before_diagnostics),
            jax.tree_util.tree_leaves(potential),
        )
    )

    utd_batch = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value), axis=0), batch
    )
    step_before = potential.network.step
    potential, batch_info = potential.batch_update(utd_batch, world_model)
    assert potential.network.step == step_before + 2
    assert bool(batch_info["is_finite"])

    print("latent potential smoke test passed")
    print(f"initial_loss: {float(initial_info['loss']):.6f}")
    print(f"final_loss: {float(final_info['loss']):.6f}")
    print(f"correlation: {float(final_info['correlation']):.6f}")
    print(f"r2: {float(final_info['r2']):.6f}")


if __name__ == "__main__":
    main()
