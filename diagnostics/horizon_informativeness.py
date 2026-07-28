"""Measure control-target informativeness across real dataset horizons."""

import argparse
import json

import gymnasium
import numpy as np
import ogbench  # noqa: F401  Registers OGBench environments.


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.mean(left**2) * np.mean(right**2))
    if denominator <= 0:
        return None
    return float(np.mean(left * right) / denominator)


def terminal_valid_starts(terminals, horizon):
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    if horizon <= 0 or horizon >= len(terminals):
        raise ValueError("horizon must be positive and smaller than the dataset")
    starts = np.arange(len(terminals) - horizon, dtype=np.int64)
    terminal_prefix = np.concatenate(
        ([0], np.cumsum(terminals, dtype=np.int64))
    )
    terminal_counts = (
        terminal_prefix[starts + horizon] - terminal_prefix[starts]
    )
    return starts[terminal_counts == 0]


def masked_mean(values, mask):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.asarray(values)[mask]))


def horizon_metrics(
    terminals,
    progress,
    potentials,
    full_success,
    horizon,
    delta_threshold=0.01,
):
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    progress = np.asarray(progress).reshape(-1)
    potentials = np.asarray(potentials).reshape(-1)
    full_success = np.asarray(full_success, dtype=bool).reshape(-1)
    size = len(terminals)
    if not (
        len(progress) == size
        and len(potentials) == size
        and len(full_success) == size
    ):
        raise ValueError("all diagnostic arrays must have the same length")

    terminal_indices = np.flatnonzero(terminals)
    if len(terminal_indices) == 0 or terminal_indices[-1] != size - 1:
        raise ValueError("the raw dataset must end at an episode terminal")

    starts = terminal_valid_starts(terminals, horizon)
    ends = starts + horizon
    episode_positions = np.searchsorted(
        terminal_indices, starts, side="left"
    )
    episode_end_indices = terminal_indices[episode_positions]
    episode_final_success = full_success[episode_end_indices]
    episode_final_progress = progress[episode_end_indices]
    episode_start_indices = np.concatenate(
        ([0], terminal_indices[:-1] + 1)
    )
    episode_peak_progress_values = np.maximum.reduceat(
        progress, episode_start_indices
    )
    episode_peak_progress = episode_peak_progress_values[
        episode_positions
    ]
    success_prefix = np.concatenate(
        ([0], np.cumsum(full_success, dtype=np.int64))
    )
    horizon_success = (
        success_prefix[ends + 1] - success_prefix[starts + 1]
    ) > 0

    progress_delta = progress[ends] - progress[starts]
    potential_delta = potentials[ends] - potentials[starts]
    progress_decrease = progress_delta < 0
    progress_neutral = progress_delta == 0
    progress_increase = progress_delta > 0
    potential_decrease = potential_delta < -delta_threshold
    potential_neutral = np.abs(potential_delta) <= delta_threshold
    potential_increase = potential_delta > delta_threshold

    result = {
        "horizon": int(horizon),
        "candidate_start_count": int(size - horizon),
        "valid_start_count": int(len(starts)),
        "valid_start_fraction": float(len(starts) / (size - horizon)),
        "progress_delta_mean": float(np.mean(progress_delta)),
        "progress_delta_std": float(np.std(progress_delta)),
        "progress_delta_abs_mean": float(np.mean(np.abs(progress_delta))),
        "progress_nonzero_fraction": float(
            np.mean(progress_delta != 0)
        ),
        "progress_decrease_fraction": float(np.mean(progress_decrease)),
        "progress_neutral_fraction": float(np.mean(progress_neutral)),
        "progress_increase_fraction": float(np.mean(progress_increase)),
        "potential_delta_mean": float(np.mean(potential_delta)),
        "potential_delta_std": float(np.std(potential_delta)),
        "potential_delta_abs_mean": float(
            np.mean(np.abs(potential_delta))
        ),
        "potential_decrease_fraction": float(
            np.mean(potential_decrease)
        ),
        "potential_neutral_fraction": float(np.mean(potential_neutral)),
        "potential_increase_fraction": float(
            np.mean(potential_increase)
        ),
        "end_success_fraction": float(np.mean(full_success[ends])),
        "horizon_success_fraction": float(np.mean(horizon_success)),
        "episode_final_success_fraction": float(
            np.mean(episode_final_success)
        ),
        "episode_final_progress_mean": float(
            np.mean(episode_final_progress)
        ),
        "episode_final_progress_std": float(
            np.std(episode_final_progress)
        ),
        "episode_peak_progress_mean": float(
            np.mean(episode_peak_progress)
        ),
        "episode_peak_progress_std": float(
            np.std(episode_peak_progress)
        ),
        "progress_delta_horizon_success_correlation": safe_correlation(
            progress_delta, horizon_success
        ),
        "potential_delta_horizon_success_correlation": safe_correlation(
            potential_delta, horizon_success
        ),
        "progress_delta_episode_final_success_correlation": safe_correlation(
            progress_delta, episode_final_success
        ),
        "potential_delta_episode_final_success_correlation": safe_correlation(
            potential_delta, episode_final_success
        ),
        "progress_delta_episode_final_progress_correlation": safe_correlation(
            progress_delta, episode_final_progress
        ),
        "potential_delta_episode_final_progress_correlation": safe_correlation(
            potential_delta, episode_final_progress
        ),
        "progress_delta_episode_peak_progress_correlation": safe_correlation(
            progress_delta, episode_peak_progress
        ),
        "potential_delta_episode_peak_progress_correlation": safe_correlation(
            potential_delta, episode_peak_progress
        ),
    }
    for label, mask in (
        ("decrease", potential_decrease),
        ("neutral", potential_neutral),
        ("increase", potential_increase),
    ):
        result[
            f"episode_final_success_given_potential_{label}"
        ] = masked_mean(episode_final_success, mask)
        result[
            f"horizon_success_given_potential_{label}"
        ] = masked_mean(horizon_success, mask)
        result[
            f"episode_final_progress_given_potential_{label}"
        ] = masked_mean(episode_final_progress, mask)
        result[
            f"episode_peak_progress_given_potential_{label}"
        ] = masked_mean(episode_peak_progress, mask)
    return result


def parse_horizons(value):
    horizons = tuple(int(item.strip()) for item in value.split(","))
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise argparse.ArgumentTypeError(
            "horizons must be a comma-separated list of positive integers"
        )
    if len(set(horizons)) != len(horizons):
        raise argparse.ArgumentTypeError("horizons must be unique")
    return horizons


def resolve_ogbench_env_name(dataset_env_name):
    """Remove the OGBench dataset-type token from an environment config name."""
    splits = dataset_env_name.split("-")
    if "singletask" not in splits:
        return dataset_env_name
    position = splits.index("singletask")
    if position < 1:
        raise ValueError("singletask environment name is malformed")
    return "-".join(splits[: position - 1] + splits[position:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="/root/.ogbench/data/cube-triple-play-v0.npz",
    )
    parser.add_argument(
        "--env_name",
        default="cube-triple-play-singletask-task2-v0",
    )
    parser.add_argument("--horizons", type=parse_horizons, default=(5, 10, 20, 40))
    parser.add_argument("--success_tolerance", type=float, default=0.04)
    parser.add_argument("--potential_temperature", type=float, default=0.04)
    parser.add_argument("--delta_threshold", type=float, default=0.01)
    args = parser.parse_args()
    if args.success_tolerance <= 0:
        raise ValueError("success_tolerance must be positive")
    if args.potential_temperature <= 0:
        raise ValueError("potential_temperature must be positive")
    if args.delta_threshold <= 0:
        raise ValueError("delta_threshold must be positive")

    with np.load(args.dataset_path) as dataset_file:
        qpos = dataset_file["qpos"]
        terminals = dataset_file["terminals"].astype(bool)

    resolved_env_name = resolve_ogbench_env_name(args.env_name)
    env = gymnasium.make(resolved_env_name)
    try:
        env.reset()
        num_cubes = env.unwrapped._num_cubes
        target_cube_xyzs = env.unwrapped._data.mocap_pos.copy()[:num_cubes]
    finally:
        env.close()

    cube_xyzs = np.stack(
        [
            qpos[:, 14 + 7 * index : 17 + 7 * index]
            for index in range(num_cubes)
        ],
        axis=1,
    )
    distances = np.linalg.norm(
        cube_xyzs - target_cube_xyzs[None], axis=-1
    )
    cube_successes = distances <= args.success_tolerance
    progress = cube_successes.sum(axis=-1)
    full_success = np.all(cube_successes, axis=-1)
    scaled_distances = (
        distances - args.success_tolerance
    ) / args.potential_temperature
    potentials = (
        1.0
        / (1.0 + np.exp(np.clip(scaled_distances, -60.0, 60.0)))
    ).sum(axis=-1)

    results = {
        "metadata": {
            "dataset_path": args.dataset_path,
            "env_name": args.env_name,
            "resolved_env_name": resolved_env_name,
            "state_count": int(len(terminals)),
            "episode_count": int(np.sum(terminals)),
            "num_cubes": int(num_cubes),
            "success_tolerance": args.success_tolerance,
            "potential_temperature": args.potential_temperature,
            "delta_threshold": args.delta_threshold,
            "target_cube_xyzs": target_cube_xyzs.tolist(),
            "full_success_state_count": int(np.sum(full_success)),
        },
        "horizons": {
            str(horizon): horizon_metrics(
                terminals,
                progress,
                potentials,
                full_success,
                horizon,
                delta_threshold=args.delta_threshold,
            )
            for horizon in args.horizons
        },
    }

    print("Real-data horizon informativeness diagnostic:")
    print(
        "  H  valid_starts  progress_nonzero  potential_std  "
        "potential_dec/neutral/inc  final_success_corr  final_progress_corr"
    )
    for horizon in args.horizons:
        metrics = results["horizons"][str(horizon)]
        final_success_correlation = metrics[
            "potential_delta_episode_final_success_correlation"
        ]
        final_success_text = (
            "n/a"
            if final_success_correlation is None
            else f"{final_success_correlation:.6f}"
        )
        print(
            f"  {horizon:>2}  {metrics['valid_start_count']:>12}  "
            f"{metrics['progress_nonzero_fraction']:.6f}  "
            f"{metrics['potential_delta_std']:.6f}  "
            f"{metrics['potential_decrease_fraction']:.6f}/"
            f"{metrics['potential_neutral_fraction']:.6f}/"
            f"{metrics['potential_increase_fraction']:.6f}  "
            f"{final_success_text:>8}  "
            f"{metrics['potential_delta_episode_final_progress_correlation']:.6f}"
        )
    print("JSON_RESULT=" + json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
