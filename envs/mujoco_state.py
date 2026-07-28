"""Exact, in-process snapshots for evaluation-only MuJoCo branching.

The snapshot is intentionally independent of training and action selection.  It
captures the MuJoCo integration state together with mutable Gymnasium wrapper
and OGBench task state so callers can run a branch and then restore the main
episode.
"""

import contextlib
import copy
import random
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


_CORE_STATE_ATTRIBUTES = (
    "_reset_next_step",
    "_prev_qpos",
    "_prev_qvel",
    "_prev_ob_info",
    "_success",
    "_target_effector_pose",
    "_target_block",
    "_target_task",
    "cur_task_id",
    "cur_task_info",
    "_cur_goal_ob",
    "_cur_goal_rendered",
    "_render_goal",
)

_WRAPPER_STATE_ATTRIBUTES = (
    # Gymnasium TimeLimit / order and environment checkers.
    "_elapsed_steps",
    "_has_reset",
    "checked_reset",
    "checked_step",
    "checked_render",
    "close_called",
    # QC EpisodeMonitor.
    "reward_sum",
    "episode_length",
    "total_timesteps",
    "start_time",
    # QC FrameStackWrapper.
    "frames",
)


@dataclass(frozen=True)
class _ObjectState:
    """Selected mutable attributes belonging to one live object."""

    owner: Any
    attributes: dict[str, Any]


@dataclass(frozen=True)
class _GeneratorState:
    """State of one NumPy Generator without replacing the live object."""

    generator: np.random.Generator
    bit_generator_state: dict[str, Any]


@dataclass(frozen=True)
class MujocoEnvSnapshot:
    """In-process snapshot of a wrapped OGBench MuJoCo environment."""

    core: Any
    model_signature: tuple[int, int, int, int, int]
    mj_data: Any
    integration_state: np.ndarray
    geom_rgba: np.ndarray
    object_states: tuple[_ObjectState, ...]
    generator_states: tuple[_GeneratorState, ...]
    numpy_random_state: tuple[Any, ...]
    python_random_state: object


def _wrapper_chain(env):
    """Return every wrapper followed by the unwrapped environment."""
    chain = []
    current = env
    seen = set()
    while id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        if not hasattr(current, "env"):
            break
        current = current.env
    return chain


def _selected_state(owner, attribute_names):
    attributes = {
        name: copy.deepcopy(getattr(owner, name))
        for name in attribute_names
        if hasattr(owner, name)
    }
    return _ObjectState(owner=owner, attributes=attributes)


def _collect_generators(objects):
    """Collect distinct persistent environment and wrapper RNGs."""
    generators = []
    seen = set()
    for owner in objects:
        for name in ("_np_random", "np_random"):
            try:
                generator = getattr(owner, name)
            except (AttributeError, RuntimeError):
                continue
            if (
                isinstance(generator, np.random.Generator)
                and id(generator) not in seen
            ):
                seen.add(id(generator))
                generators.append(
                    _GeneratorState(
                        generator=generator,
                        bit_generator_state=copy.deepcopy(
                            generator.bit_generator.state
                        ),
                    )
                )
    return tuple(generators)


def clone_mujoco_env_state(env):
    """Clone a wrapped MuJoCo environment for deterministic branch rollout.

    Snapshots are in-process objects: they retain references to the exact live
    environment and RNG objects and must be restored into that same environment.
    """
    chain = _wrapper_chain(env)
    core = env.unwrapped
    if not hasattr(core, "model") or not hasattr(core, "data"):
        raise TypeError("Environment does not expose MuJoCo model/data")

    model = core.model
    data = core.data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    integration_state = np.empty(
        mujoco.mj_stateSize(model, state_spec), dtype=np.float64
    )
    mujoco.mj_getState(model, data, integration_state, state_spec)

    object_states = [
        _selected_state(wrapper, _WRAPPER_STATE_ATTRIBUTES)
        for wrapper in chain[:-1]
    ]
    object_states.append(_selected_state(core, _CORE_STATE_ATTRIBUTES))

    return MujocoEnvSnapshot(
        core=core,
        model_signature=(model.nq, model.nv, model.na, model.nu, model.nmocap),
        # A complete MjData copy includes solver/contact caches that are
        # outside mjSTATE_INTEGRATION but affect exact continuation.
        mj_data=copy.copy(data),
        integration_state=integration_state,
        geom_rgba=np.array(model.geom_rgba, copy=True),
        object_states=tuple(object_states),
        generator_states=_collect_generators(chain),
        numpy_random_state=copy.deepcopy(np.random.get_state()),
        python_random_state=copy.deepcopy(random.getstate()),
    )


def restore_mujoco_env_state(env, snapshot):
    """Restore a snapshot without resetting or advancing the environment."""
    core = env.unwrapped
    if core is not snapshot.core:
        raise ValueError("Snapshot belongs to a different environment instance")

    model = core.model
    signature = (model.nq, model.nv, model.na, model.nu, model.nmocap)
    if signature != snapshot.model_signature:
        raise ValueError("MuJoCo model shape changed since the snapshot")

    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    expected_size = mujoco.mj_stateSize(model, state_spec)
    if snapshot.integration_state.shape != (expected_size,):
        raise ValueError("Snapshot integration-state size is incompatible")

    # Use a fresh copy so repeated restores cannot mutate the snapshot.
    # Calling mj_forward here would overwrite restored solver caches.
    core._data = copy.copy(snapshot.mj_data)
    model.geom_rgba[:] = snapshot.geom_rgba
    for object_state in snapshot.object_states:
        for name, value in object_state.attributes.items():
            setattr(object_state.owner, name, copy.deepcopy(value))
    for generator_state in snapshot.generator_states:
        generator_state.generator.bit_generator.state = copy.deepcopy(
            generator_state.bit_generator_state
        )
    np.random.set_state(copy.deepcopy(snapshot.numpy_random_state))
    random.setstate(copy.deepcopy(snapshot.python_random_state))


@contextlib.contextmanager
def preserved_mujoco_env_state(env):
    """Restore the main episode even if a diagnostic branch raises."""
    snapshot = clone_mujoco_env_state(env)
    try:
        yield snapshot
    finally:
        restore_mujoco_env_state(env, snapshot)
