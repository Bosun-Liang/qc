"""Smoke test for the class-balanced latent task-progress head."""

import jax
import jax.numpy as jnp

from agents.acfql import ACFQLAgent, get_config
from agents.world_model import LatentProgressTrainState, WorldModelTrainState


def main():
    batch_size = 32
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
    reward_classes = jnp.tile(
        jnp.asarray([-3.0, -2.0, -1.0, 0.0]), batch_size // 4
    )
    rewards = jnp.repeat(
        reward_classes[:, None], horizon_length, axis=1
    )
    batch = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
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
    progress = LatentProgressTrainState.create(
        seed=2,
        latent_dim=16,
        hidden_dims=(32, 32),
        class_weights=(0.25, 0.5, 1.0, 1.0),
    )
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
    world_model_before = jax.tree_util.tree_map(jnp.copy, world_model)
    initial_info = progress.evaluate(batch, world_model)

    for _ in range(20):
        progress, info = progress.update(batch, world_model)
        assert bool(info["is_finite"])
        assert bool(jnp.isfinite(info["grad/norm"]))

    final_info = progress.evaluate(batch, world_model)
    assert float(final_info["loss"]) < float(initial_info["loss"])
    assert 0.0 <= float(final_info["accuracy"]) <= 1.0
    assert 0.0 <= float(final_info["balanced_accuracy"]) <= 1.0
    for class_index in range(4):
        assert 0.0 <= float(
            final_info[f"class_{class_index}_recall"]
        ) <= 1.0
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
        "rewards": reward_classes,
    }
    transition_info = progress.evaluate(transition_batch, world_model)
    assert bool(transition_info["is_finite"])

    progress_before_candidates = jax.tree_util.tree_map(jnp.copy, progress)
    future_info = progress.evaluate_predicted_future(
        {
            "observations": observations,
            "actions": actions,
            "target_observations": next_observations[:, -1],
            "target_rewards": reward_classes,
        },
        world_model,
    )
    assert bool(future_info["is_finite"])
    assert 0.0 <= float(future_info["accuracy"]) <= 1.0
    assert 0.0 <= float(future_info["balanced_accuracy"]) <= 1.0
    assert -1.0 <= float(future_info["correlation"]) <= 1.0
    assert -1.0 <= float(
        future_info["score_vs_encoded_correlation"]
    ) <= 1.0
    assert -1.0 <= float(future_info["latent_cosine"]) <= 1.0

    candidate_info = progress.evaluate_candidates(
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
        candidate_info["critic_choice_progress_percentile"]
    ) <= 1.0
    assert 0.0 <= float(
        candidate_info["progress_choice_critic_percentile"]
    ) <= 1.0
    assert all(
        bool(jnp.isfinite(value)) for value in candidate_info.values()
    )
    assert all(
        bool(jnp.array_equal(before, after))
        for before, after in zip(
            jax.tree_util.tree_leaves(progress_before_candidates),
            jax.tree_util.tree_leaves(progress),
        )
    )

    utd_batch = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value), axis=0), batch
    )
    step_before = progress.network.step
    progress, batch_info = progress.batch_update(utd_batch, world_model)
    assert progress.network.step == step_before + 2
    assert bool(batch_info["is_finite"])

    print("latent progress smoke test passed")
    print(f"initial_loss: {float(initial_info['loss']):.6f}")
    print(f"final_loss: {float(final_info['loss']):.6f}")
    print(f"accuracy: {float(final_info['accuracy']):.6f}")
    print(
        "balanced_accuracy: "
        f"{float(final_info['balanced_accuracy']):.6f}"
    )


if __name__ == "__main__":
    main()
