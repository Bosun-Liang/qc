"""Minimal forward, loss, gradient, and JIT smoke test for the world model."""

import jax
import jax.numpy as jnp

from agents.world_model import (
    WorldModel,
    WorldModelTrainState,
    world_model_loss,
)


def main():
    batch_size = 4
    horizon_length = 5
    observation_dim = 46
    action_dim = 5

    observation_rng, action_rng, next_observation_rng, init_rng = (
        jax.random.split(jax.random.PRNGKey(0), 4)
    )
    observations = jax.random.normal(
        observation_rng,
        (batch_size, observation_dim),
        dtype=jnp.float32,
    )
    actions = jax.random.normal(
        action_rng,
        (batch_size, horizon_length, action_dim),
        dtype=jnp.float32,
    )
    next_observations = jax.random.normal(
        next_observation_rng,
        (batch_size, horizon_length, observation_dim),
        dtype=jnp.float32,
    )
    valid = jnp.ones(
        (batch_size, horizon_length),
        dtype=jnp.float32,
    ).at[-1, -1].set(0.0)
    batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "valid": valid,
    }

    model = WorldModel(
        latent_dim=64,
        hidden_dims=(256, 256),
    )
    params = model.init(
        init_rng,
        observations,
        actions,
        next_observations[..., -1, :],
    )["params"]

    def loss_fn(model_params, model_batch):
        return world_model_loss(model.apply, model_params, model_batch)

    jitted_loss_fn = jax.jit(loss_fn)
    loss, info = jitted_loss_fn(params, batch)
    grads = jax.grad(lambda p: loss_fn(p, batch)[0])(params)

    changed_batch = dict(batch)
    changed_batch["next_observations"] = next_observations.at[
        -1, -1
    ].set(1e6)
    changed_loss, _ = jitted_loss_fn(params, changed_batch)

    assert loss.shape == ()
    assert bool(info["is_finite"])
    assert float(loss) > 0.0
    assert float(info["latent_std"]) > 0.0
    assert bool(jnp.isclose(loss, changed_loss))
    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(grads)
    )

    state = WorldModelTrainState.create(
        seed=1,
        example_observations=observations[0],
        example_actions=actions[0, 0],
        horizon_length=horizon_length,
        latent_dim=64,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        coef=0.1,
    )
    first_update_loss = None
    for _ in range(10):
        state, update_info = state.update(batch)
        if first_update_loss is None:
            first_update_loss = update_info["loss"]
        assert bool(update_info["is_finite"])
        assert bool(jnp.isfinite(update_info["grad/norm"]))

    assert state.network.step == 11
    assert float(update_info["loss"]) < float(first_update_loss)

    utd_batch = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value), axis=0),
        batch,
    )
    state, batch_update_info = state.batch_update(utd_batch)
    assert state.network.step == 13
    assert bool(batch_update_info["is_finite"])
    assert bool(jnp.isfinite(batch_update_info["grad/norm"]))

    print("world model smoke test passed")
    print(f"loss: {float(loss):.6f}")
    print(f"latent_std: {float(info['latent_std']):.6f}")
    print(f"valid_fraction: {float(info['valid_fraction']):.6f}")
    print(f"loss_after_10_updates: {float(update_info['loss']):.6f}")


if __name__ == "__main__":
    main()
