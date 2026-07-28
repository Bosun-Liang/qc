"""Smoke tests for low-capacity selector models and learning-curve subsets."""

import tempfile
from pathlib import Path

import jax
import numpy as np

from diagnostics.low_capacity_selector import (
    MODEL_NAMES,
    PARAMETER_COUNTS,
    build_nested_episode_subsets,
    create_low_capacity_state,
    low_capacity_evaluate_objective,
    low_capacity_predict,
    low_capacity_train_step,
    normalized_state_inputs,
)
from diagnostics.pairwise_selector import (
    compute_normalization,
    load_and_validate_dataset,
    restore_state,
    save_state,
)


DATASET = "/root/autodl-tmp/qc_workspace/datasets/qc_branch_preferences_h80_seed0.npz"


def assert_trees_close(left, right, rtol=0, atol=0):
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_value, right_value in zip(left_leaves, right_leaves):
        np.testing.assert_allclose(
            np.asarray(left_value),
            np.asarray(right_value),
            rtol=rtol,
            atol=atol,
        )


def main():
    _, grouped = load_and_validate_dataset(DATASET)
    validation_ids = np.array(grouped["validation"].state_ids, copy=True)
    test_ids = np.array(grouped["test"].state_ids, copy=True)
    subsets, metadata = build_nested_episode_subsets(grouped["train"], seed=0)
    assert [len(subsets[target].state_ids) for target in (25, 50, 100, 140)] == [
        30,
        50,
        100,
        140,
    ]
    previous = set()
    for target in (25, 50, 100, 140):
        state_ids = set(int(value) for value in subsets[target].state_ids)
        assert previous.issubset(state_ids)
        previous = state_ids
        assert len(np.unique(subsets[target].episode_ids)) == metadata[target]["episodes"]
        normalization = compute_normalization(subsets[target])
        observations, actions = normalized_state_inputs(
            subsets[target], normalization
        )
        assert observations.shape == (len(state_ids), 8, 46)
        assert actions.shape == (len(state_ids), 8, 25)
        assert np.isfinite(observations).all()
        assert np.isfinite(actions).all()
        np.testing.assert_array_equal(
            validation_ids, grouped["validation"].state_ids
        )
        np.testing.assert_array_equal(test_ids, grouped["test"].state_ids)

    normalization = compute_normalization(subsets[25])
    observations, actions = normalized_state_inputs(subsets[25], normalization)
    for model_name in MODEL_NAMES:
        first = create_low_capacity_state(model_name, seed=0)
        second = create_low_capacity_state(model_name, seed=0)
        assert_trees_close(first.params, second.params)
        parameter_count = int(
            sum(
                np.prod(value.shape)
                for value in jax.tree_util.tree_leaves(first.params)
            )
        )
        assert parameter_count == PARAMETER_COUNTS[model_name]
        scores = np.asarray(
            low_capacity_predict(first, observations[:4], actions[:4])
        )
        assert scores.shape == (4, 8)
        assert np.isfinite(scores).all()
        first, first_metrics = low_capacity_train_step(
            first,
            observations[:4],
            actions[:4],
            subsets[25].returns[:4],
            np.float32(1.0),
        )
        second, second_metrics = low_capacity_train_step(
            second,
            observations[:4],
            actions[:4],
            subsets[25].returns[:4],
            np.float32(1.0),
        )
        assert_trees_close(first.params, second.params, rtol=1e-7, atol=1e-9)
        for metrics in (first_metrics, second_metrics):
            assert bool(np.asarray(metrics["gradient_finite"]))
            assert np.isfinite(np.asarray(metrics["loss"]))
            assert np.isfinite(np.asarray(metrics["grad_norm"]))
        evaluation = low_capacity_evaluate_objective(
            first,
            observations[:4],
            actions[:4],
            subsets[25].returns[:4],
            np.float32(1.0),
        )
        assert np.isfinite(np.asarray(evaluation["loss"]))
        trained_scores = np.asarray(
            low_capacity_predict(first, observations[:4], actions[:4])
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / f"{model_name}.pkl"
            save_state(checkpoint, first)
            restored = restore_state(
                checkpoint, create_low_capacity_state(model_name, seed=0)
            )
            np.testing.assert_array_equal(
                trained_scores,
                np.asarray(
                    low_capacity_predict(
                        restored, observations[:4], actions[:4]
                    )
                ),
            )
    print("learning_curve_subset_leakage_check=passed")
    print("nested_subset_check=passed actual_states=[30,50,100,140]")
    print("validation_test_fixed_check=passed")
    print("bilinear_forward_update=passed")
    print("small_mlp_forward_update=passed")
    print("linear_action_forward_update=passed")
    print("finite_loss_and_gradients=passed")
    print("checkpoint_save_restore=passed")
    print("deterministic_seed_smoke=passed")
    print("low_capacity_selector_smoke=passed")


if __name__ == "__main__":
    main()
