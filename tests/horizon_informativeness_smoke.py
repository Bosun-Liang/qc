"""Smoke test for terminal-safe horizon informativeness metrics."""

import numpy as np

from diagnostics.horizon_informativeness import (
    horizon_metrics,
    resolve_ogbench_env_name,
    safe_correlation,
    terminal_valid_starts,
)


def main():
    assert safe_correlation(np.ones(4), np.arange(4)) is None
    assert (
        resolve_ogbench_env_name(
            "cube-triple-play-singletask-task2-v0"
        )
        == "cube-triple-singletask-task2-v0"
    )
    terminals = np.asarray(
        [False, False, False, False, False, True] * 2
    )
    progress = np.asarray([0, 0, 1, 1, 2, 3, 0, 0, 0, 1, 2, 2])
    potentials = progress.astype(np.float32) / 3.0
    full_success = progress == 3

    starts = terminal_valid_starts(terminals, horizon=2)
    assert np.array_equal(starts, np.asarray([0, 1, 2, 3, 6, 7, 8, 9]))
    metrics = horizon_metrics(
        terminals,
        progress,
        potentials,
        full_success,
        horizon=2,
        delta_threshold=0.01,
    )
    assert metrics["valid_start_count"] == 8
    assert 0.0 <= metrics["progress_nonzero_fraction"] <= 1.0
    assert 0.0 <= metrics["potential_neutral_fraction"] <= 1.0
    assert 0.0 <= metrics["episode_final_success_fraction"] <= 1.0
    assert np.isfinite(
        metrics["potential_delta_episode_final_success_correlation"]
    )
    assert np.isfinite(
        metrics["potential_delta_episode_final_progress_correlation"]
    )
    print("horizon informativeness smoke test passed")


if __name__ == "__main__":
    main()
