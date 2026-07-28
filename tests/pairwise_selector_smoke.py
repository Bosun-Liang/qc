"""Smoke tests for the independent Pairwise Selector v1 pipeline."""

import tempfile
from pathlib import Path

import jax
import numpy as np

from diagnostics.pairwise_selector import (
    compute_normalization,
    create_train_state,
    evaluate_objective,
    load_and_validate_dataset,
    normalized_inputs,
    pair_statistics,
    predict_scores,
    ranking_metrics,
    restore_state,
    save_state,
    train_step,
)


DATASET = "/root/autodl-tmp/qc_workspace/datasets/qc_branch_preferences_h80_seed0.npz"


def assert_trees_equal(left, right, *, rtol=0, atol=0):
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
    ):
        np.testing.assert_allclose(
            np.asarray(left_leaf), np.asarray(right_leaf), rtol=rtol, atol=atol
        )


def main():
    arrays, grouped = load_and_validate_dataset(DATASET)
    normalization = compute_normalization(grouped["train"])
    full_inputs = normalized_inputs(grouped["train"], normalization, "full")
    observation_inputs = normalized_inputs(
        grouped["train"], normalization, "observation_only"
    )
    assert full_inputs.shape == (140, 8, 71)
    assert observation_inputs.shape == (140, 8, 46)

    first = create_train_state("full", seed=0)
    second = create_train_state("full", seed=0)
    assert_trees_equal(first.params, second.params)
    before_scores = np.asarray(predict_scores(first, full_inputs[:4]))
    assert before_scores.shape == (4, 8)
    assert np.isfinite(before_scores).all()

    first, first_metrics = train_step(
        first, full_inputs[:4], grouped["train"].returns[:4], np.float32(1.0)
    )
    second, second_metrics = train_step(
        second, full_inputs[:4], grouped["train"].returns[:4], np.float32(1.0)
    )
    assert_trees_equal(first.params, second.params, rtol=1e-7, atol=1e-9)
    for metrics in (first_metrics, second_metrics):
        assert bool(np.asarray(metrics["gradient_finite"]))
        assert np.isfinite(np.asarray(metrics["loss"]))
        assert np.isfinite(np.asarray(metrics["grad_norm"]))

    evaluation = evaluate_objective(
        first, full_inputs[:4], grouped["train"].returns[:4], np.float32(1.0)
    )
    assert np.isfinite(np.asarray(evaluation["loss"]))
    after_scores = np.asarray(predict_scores(first, full_inputs[:4]))
    assert np.isfinite(after_scores).all()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "selector.pkl"
        save_state(checkpoint, first)
        restored = restore_state(checkpoint, create_train_state("full", seed=0))
        restored_scores = np.asarray(predict_scores(restored, full_inputs[:4]))
        np.testing.assert_array_equal(after_scores, restored_scores)
        assert_trees_equal(first.params, restored.params)

    observation_state = create_train_state("observation_only", seed=0)
    observation_scores = np.asarray(
        predict_scores(observation_state, observation_inputs[:4])
    )
    np.testing.assert_array_equal(
        observation_scores,
        np.repeat(observation_scores[:, :1], 8, axis=1),
    )
    observation_metrics = ranking_metrics(
        grouped["train"].returns[:4], observation_scores
    )
    assert observation_metrics["pairwise_accuracy"] == 0.5
    assert observation_metrics["mean_score_spread"] == 0.0

    stats = pair_statistics(grouped)
    assert stats["train"]["states"] == 140
    assert stats["validation"]["states"] == 30
    assert stats["test"]["states"] == 30
    assert arrays["candidate_action_chunk"].shape == (1600, 5, 5)
    assert np.array_equal(arrays["reward_mask"].sum(axis=1), arrays["executed_steps"])
    print("dataset_schema_validation=passed")
    print("single_batch_forward=passed")
    print("single_step_update=passed")
    print("jit_smoke=passed")
    print("loss_and_gradient_finite=passed")
    print("checkpoint_save_restore=passed")
    print("restore_prediction_exact=passed")
    print("deterministic_seed_update=passed")
    print("observation_only_identical_scores=passed")
    print("pairwise_selector_smoke=passed")


if __name__ == "__main__":
    main()
