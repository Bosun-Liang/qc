"""Smoke test for the diagnostic latent future-value head."""

import jax
import jax.numpy as jnp

from agents.acfql import ACFQLAgent, get_config
from agents.world_model import LatentValueTrainState, WorldModelTrainState


def main():
    batch_size = 16
    horizon_length = 5
    observation_dim = 46
    action_dim = 5

    observation_rng, action_rng, next_observation_rng = jax.random.split(
        jax.random.PRNGKey(0),
        3,
    )
    observations = jax.random.normal(
        observation_rng,
        (batch_size, observation_dim),
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
    batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "valid": jnp.ones((batch_size, horizon_length)),
    }

    config = get_config()
    config.horizon_length = horizon_length
    config.actor_type = "best-of-n"
    config.actor_num_samples = 4
    config.actor_hidden_dims = (32, 32)
    config.value_hidden_dims = (32, 32)
    config.flow_steps = 2
    agent = ACFQLAgent.create(
        0,
        observations[0],
        actions[0, 0],
        config,
    )
    world_model = WorldModelTrainState.create(
        seed=1,
        example_observations=observations[0],
        example_actions=actions[0, 0],
        horizon_length=horizon_length,
        latent_dim=16,
        hidden_dims=(32, 32),
    )
    value_state = LatentValueTrainState.create(
        seed=2,
        latent_dim=16,
        hidden_dims=(32, 32),
    )

    agent_before = jax.tree_util.tree_map(jnp.copy, agent)
    world_model_before = jax.tree_util.tree_map(jnp.copy, world_model)

    for _ in range(10):
        value_state, info = value_state.update(batch, world_model, agent)
        assert bool(info["is_finite"])
        assert bool(jnp.isfinite(info["grad/norm"]))

    assert value_state.network.step == 11
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(agent_before),
            jax.tree_util.tree_leaves(agent),
        )
    )
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(world_model_before),
            jax.tree_util.tree_leaves(world_model),
        )
    )

    eval_info = value_state.evaluate(batch, world_model, agent)
    assert bool(eval_info["is_finite"])
    assert 0.0 <= float(eval_info["pairwise_agreement"]) <= 1.0
    assert 0.0 <= float(eval_info["topk_agreement"]) <= 1.0

    utd_batch = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value), axis=0),
        batch,
    )
    value_state, batch_info = value_state.batch_update(
        utd_batch,
        world_model,
        agent,
    )
    assert value_state.network.step == 13
    assert bool(batch_info["is_finite"])

    print("latent value smoke test passed")
    print(f"loss: {float(eval_info['loss']):.6f}")
    print(f"correlation: {float(eval_info['correlation']):.6f}")
    print(
        "pairwise_agreement: "
        f"{float(eval_info['pairwise_agreement']):.6f}"
    )
    print(f"topk_agreement: {float(eval_info['topk_agreement']):.6f}")


if __name__ == "__main__":
    main()
