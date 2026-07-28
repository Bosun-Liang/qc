import glob, tqdm, wandb, os, json, random, time, jax
import functools
from absl import app, flags
from ml_collections import config_flags
from log_utils import setup_wandb, get_exp_name, get_flag_dict, CsvLogger, get_wandb_video

from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets
def is_robomimic_env(env_name):
    """Return whether env_name refers to a supported RoboMimic task."""
    if "low_dim" not in env_name:
        return False

    parts = env_name.split("-")
    if len(parts) != 3:
        return False

    task, dataset_type, _ = parts
    return (
        task in ("lift", "can", "square", "transport", "tool_hang")
        and dataset_type in ("mh", "ph")
    )


from utils.flax_utils import save_agent, restore_agent_with_file
from utils.datasets import Dataset, ReplayBuffer

from evaluation import evaluate, flatten
from agents import agents
from agents.world_model import (
    LatentPotentialDeltaTrainState,
    LatentPotentialTrainState,
    LatentProgressTrainState,
    LatentValueTrainState,
    WorldModelTrainState,
)
import numpy as np

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']
    os.environ['MUJOCO_EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES']

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'cube-triple-play-singletask-task2-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_file', None, 'Checkpoint file to restore before training.')
flags.DEFINE_integer('restore_step', 0, 'Global step represented by the restored checkpoint.')
flags.DEFINE_bool('eval_only', False, 'Only evaluate the restored checkpoint and exit.')
flags.DEFINE_bool(
    'candidate_diagnostic_only',
    False,
    'Only compare critic and an enabled latent head on best-of-N candidates.',
)
flags.DEFINE_integer(
    'candidate_diagnostic_batches',
    16,
    'Number of held-out batches used by candidate_diagnostic_only.',
)
flags.DEFINE_bool(
    'wm_progress_only',
    False,
    'Freeze agent/world-model states and train only the progress head.',
)
flags.DEFINE_bool(
    'wm_potential_only',
    False,
    'Freeze agent/world-model states and train only the potential head.',
)
flags.DEFINE_bool(
    'wm_delta_only',
    False,
    'Freeze agent/world-model states and train only the potential-delta head.',
)
flags.DEFINE_bool(
    'wm_score_eval_enabled',
    False,
    'Use normalized critic plus latent-value scoring in eval-only mode.',
)
flags.DEFINE_float(
    'wm_score_lambda',
    0.0,
    'Latent-value coefficient used when wm_score_eval_enabled is true.',
)
flags.DEFINE_bool(
    "wm_delta_score_eval_enabled",
    False,
    "Use normalized critic plus latent potential-delta scoring in eval-only mode.",
)
flags.DEFINE_float(
    "wm_delta_score_lambda",
    0.0,
    "Potential-delta coefficient used when wm_delta_score_eval_enabled is true.",
)
flags.DEFINE_integer(
    "eval_seed",
    None,
    'Optional deterministic per-episode seed for eval-only comparisons.',
)

flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 1000000, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 2000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', -1, 'Save interval.')
flags.DEFINE_integer('start_training', 5000, 'when does training start')

flags.DEFINE_integer('utd_ratio', 1, "update to data ratio")

flags.DEFINE_float('discount', 0.99, 'discount factor')

flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')

config_flags.DEFINE_config_file('agent', 'agents/acfql.py', lock_config=False)

flags.DEFINE_float('dataset_proportion', 1.0, "Proportion of the dataset to use")
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('ogbench_dataset_dir', None, 'OGBench dataset directory')

flags.DEFINE_integer('horizon_length', 5, 'action chunking length.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")

flags.DEFINE_bool('save_all_online_states', False, "save all trajectories to npy")

class LoggingHelper:
    def __init__(self, csv_loggers, wandb_logger):
        self.csv_loggers = csv_loggers
        self.wandb_logger = wandb_logger
        self.first_time = time.time()
        self.last_time = time.time()

    def log(self, data, prefix, step):
        assert prefix in self.csv_loggers, prefix
        self.csv_loggers[prefix].log(data, step=step)
        self.wandb_logger.log({f'{prefix}/{k}': v for k, v in data.items()}, step=step)

def main(_):
    exp_name = get_exp_name(FLAGS.seed)
    run = setup_wandb(project='qc', group=FLAGS.run_group, name=exp_name)
    
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, FLAGS.env_name, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()

    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    config = FLAGS.agent
    if FLAGS.wm_score_eval_enabled and not FLAGS.eval_only:
        raise ValueError(
            "--wm_score_eval_enabled is restricted to --eval_only=True"
        )
    if FLAGS.wm_delta_score_eval_enabled and not FLAGS.eval_only:
        raise ValueError(
            "--wm_delta_score_eval_enabled is restricted to --eval_only=True"
        )
    if (
        FLAGS.wm_score_eval_enabled
        and FLAGS.wm_delta_score_eval_enabled
    ):
        raise ValueError(
            "--wm_score_eval_enabled and --wm_delta_score_eval_enabled are "
            "mutually exclusive"
        )
    diagnostic_only_count = sum(
        (
            FLAGS.wm_progress_only,
            FLAGS.wm_potential_only,
            FLAGS.wm_delta_only,
        )
    )
    if diagnostic_only_count > 1:
        raise ValueError(
            "wm_progress_only, wm_potential_only, and wm_delta_only are "
            "mutually exclusive"
        )
    if FLAGS.wm_progress_only:
        if FLAGS.restore_file is None:
            raise ValueError("--wm_progress_only requires --restore_file")
        if FLAGS.online_steps != 0:
            raise ValueError("--wm_progress_only requires --online_steps=0")
        if not config.get('wm_progress_enabled', False):
            raise ValueError(
                "--wm_progress_only requires wm_progress_enabled=True"
            )
    if FLAGS.wm_potential_only:
        if FLAGS.restore_file is None:
            raise ValueError("--wm_potential_only requires --restore_file")
        if FLAGS.online_steps != 0:
            raise ValueError("--wm_potential_only requires --online_steps=0")
        if not config.get('wm_potential_enabled', False):
            raise ValueError(
                "--wm_potential_only requires wm_potential_enabled=True"
            )
    if FLAGS.wm_delta_only:
        if FLAGS.restore_file is None:
            raise ValueError("--wm_delta_only requires --restore_file")
        if FLAGS.online_steps != 0:
            raise ValueError("--wm_delta_only requires --online_steps=0")
        if not config.get('wm_delta_enabled', False):
            raise ValueError(
                "--wm_delta_only requires wm_delta_enabled=True"
            )
    cube_aux_enabled = config.get(
        'wm_potential_enabled', False
    ) or config.get('wm_delta_enabled', False)
    if cube_aux_enabled:
        if FLAGS.ogbench_dataset_dir is None or 'cube' not in FLAGS.env_name:
            raise ValueError(
                "potential/delta diagnostics currently support only OGBench cube tasks"
            )
        if FLAGS.online_steps != 0:
            raise ValueError(
                "potential/delta diagnostics are restricted to online_steps=0"
            )
    
    # data loading
    if FLAGS.ogbench_dataset_dir is not None:
        # custom ogbench dataset
        assert FLAGS.dataset_proportion == 1.0
        dataset_idx = 0

        # Convert an evaluation environment name such as
        # cube-double-play-singletask-task2-v0
        # to its corresponding dataset stem:
        # cube-double-play-v0
        env_parts = FLAGS.env_name.split('-singletask-')
        if len(env_parts) == 2:
            version = env_parts[1].rsplit('-', 1)[-1]
            dataset_stem = f"{env_parts[0]}-{version}"
        else:
            dataset_stem = FLAGS.env_name

        dataset_paths = [
            file
            for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz"))
            if '-val.npz' not in file
            and os.path.basename(file).startswith(dataset_stem)
        ]

        if not dataset_paths:
            raise FileNotFoundError(
                f"No dataset matching '{dataset_stem}' was found in "
                f"{FLAGS.ogbench_dataset_dir}"
            )

        print(
            f"Matched dataset files for {FLAGS.env_name}: {dataset_paths}",
            flush=True,
        )
        env, eval_env, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
            add_info=cube_aux_enabled,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(FLAGS.env_name)

    # house keeping
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    online_rng, rng = jax.random.split(jax.random.PRNGKey(FLAGS.seed), 2)
    log_step = FLAGS.restore_step
    
    discount = FLAGS.discount
    config["horizon_length"] = FLAGS.horizon_length

    # handle dataset
    potential_target_cube_xyzs = None
    if cube_aux_enabled:
        num_cubes = env.unwrapped._num_cubes
        potential_target_cube_xyzs = (
            env.unwrapped._data.mocap_pos.copy()[:num_cubes]
        )
        if config['wm_potential_temperature'] <= 0:
            raise ValueError('wm_potential_temperature must be positive')

    def add_cube_potential_labels(ds):
        """Attach smooth cube-success labels and remove privileged fields."""
        if potential_target_cube_xyzs is None:
            return ds
        if 'qpos' not in ds:
            raise ValueError('Cube potential labels require dataset qpos')
        fields = {key: value for key, value in ds.items()}
        qpos = np.asarray(fields['qpos'])
        cube_xyzs = np.stack(
            [
                qpos[:, 14 + 7 * index : 17 + 7 * index]
                for index in range(len(potential_target_cube_xyzs))
            ],
            axis=1,
        )
        distances = np.linalg.norm(
            cube_xyzs - potential_target_cube_xyzs[None], axis=-1
        )
        scaled_distances = (
            distances - 0.04
        ) / config['wm_potential_temperature']
        smooth_successes = 1.0 / (
            1.0 + np.exp(np.clip(scaled_distances, -60.0, 60.0))
        )
        fields['wm_potentials'] = smooth_successes.sum(
            axis=-1
        ).astype(np.float32)
        for key in ('qpos', 'qvel', 'button_states'):
            fields.pop(key, None)
        return fields

    def process_train_dataset(ds):
        """
        Process the train dataset to
            - attach optional diagnostic potential labels
            - handle dataset proportion
            - handle sparse reward
        """

        ds = Dataset.create(**add_cube_potential_labels(ds))
        if FLAGS.dataset_proportion < 1.0:
            new_size = int(len(ds['masks']) * FLAGS.dataset_proportion)
            ds = Dataset.create(
                **{k: v[:new_size] for k, v in ds.items()}
            )
        
        if is_robomimic_env(FLAGS.env_name):
            penalty_rewards = ds["rewards"] - 1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = penalty_rewards
            ds = Dataset.create(**ds_dict)
        
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)

        return ds

    def get_delta_start_groups(dataset):
        """Return terminal-valid starts grouped as decrease/neutral/increase."""
        horizon_length = FLAGS.horizon_length
        max_start = dataset.size - horizon_length
        terminals = np.asarray(dataset['terminals']).reshape(-1)
        terminal_prefix = np.concatenate(
            ([0], np.cumsum(terminals > 0))
        )
        terminal_counts = (
            terminal_prefix[
                horizon_length:horizon_length + max_start
            ]
            - terminal_prefix[:max_start]
        )
        valid_starts = np.flatnonzero(terminal_counts == 0)
        potentials = np.asarray(
            dataset['wm_potentials']
        ).reshape(-1)
        deltas = (
            potentials[valid_starts + horizon_length]
            - potentials[valid_starts]
        )
        threshold = config['wm_delta_nontrivial_threshold']
        groups = [
            valid_starts[deltas < -threshold],
            valid_starts[np.abs(deltas) <= threshold],
            valid_starts[deltas > threshold],
        ]
        if any(len(group) == 0 for group in groups):
            raise ValueError(
                'Potential-delta decrease/neutral/increase groups must '
                'all be present'
            )
        return groups, deltas

    def make_delta_batch(dataset, start_indices):
        horizon_length = FLAGS.horizon_length
        action_indices = (
            start_indices[:, None]
            + np.arange(horizon_length)[None, :]
        )
        potentials = np.asarray(
            dataset['wm_potentials']
        ).reshape(-1)
        return {
            'observations': dataset['observations'][start_indices],
            'actions': dataset['actions'][action_indices],
            'wm_potential_deltas': (
                potentials[start_indices + horizon_length]
                - potentials[start_indices]
            ).astype(np.float32),
        }

    train_dataset = process_train_dataset(train_dataset)
    example_batch = train_dataset.sample(())
    
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    world_model = None
    world_model_value = None
    world_model_progress = None
    world_model_potential = None
    world_model_delta = None
    progress_train_class_indices = None
    potential_train_bin_indices = None
    delta_train_class_indices = None
    if config.get('wm_enabled', False):
        world_model = WorldModelTrainState.create(
            seed=FLAGS.seed + 1,
            example_observations=example_batch['observations'],
            example_actions=example_batch['actions'],
            horizon_length=FLAGS.horizon_length,
            latent_dim=config['wm_latent_dim'],
            hidden_dims=config['wm_hidden_dims'],
            learning_rate=config['wm_lr'],
            coef=config['wm_coef'],
        )
        print('Initialized independent auxiliary world model.', flush=True)
    if config.get('wm_value_enabled', False):
        if world_model is None:
            raise ValueError('wm_value_enabled requires wm_enabled=True')
        world_model_value = LatentValueTrainState.create(
            seed=FLAGS.seed + 2,
            latent_dim=config['wm_latent_dim'],
            hidden_dims=config['wm_value_hidden_dims'],
            learning_rate=config['wm_value_lr'],
            coef=config['wm_value_coef'],
        )
        print('Initialized diagnostic latent value head.', flush=True)
    if config.get('wm_progress_enabled', False):
        if world_model is None:
            raise ValueError('wm_progress_enabled requires wm_enabled=True')
        progress_num_classes = config['wm_progress_num_classes']
        progress_labels = np.clip(
            np.rint(np.asarray(train_dataset['rewards']) + 3.0).astype(int),
            0,
            progress_num_classes - 1,
        )
        progress_counts = np.bincount(
            progress_labels, minlength=progress_num_classes
        )
        present_counts = progress_counts[progress_counts > 0]
        if len(present_counts) == 0:
            raise ValueError('No progress labels were found in the dataset')
        progress_train_class_indices = [
            np.flatnonzero(progress_labels == class_index)
            for class_index in range(progress_num_classes)
            if progress_counts[class_index] > 0
        ]
        if FLAGS.wm_progress_only:
            # The progress-only diagnostic uses balanced sampling below, so
            # additional loss weighting would count class imbalance twice.
            progress_class_weights = np.ones(progress_num_classes)
            progress_sampling = 'balanced transition sampling'
        else:
            safe_counts = np.where(
                progress_counts > 0,
                progress_counts,
                present_counts.min(),
            )
            progress_class_weights = 1.0 / np.sqrt(
                safe_counts.astype(float)
            )
            progress_class_weights /= progress_class_weights.mean()
            progress_sampling = 'natural sequence sampling'
        world_model_progress = LatentProgressTrainState.create(
            seed=FLAGS.seed + 3,
            latent_dim=config['wm_latent_dim'],
            hidden_dims=config['wm_progress_hidden_dims'],
            learning_rate=config['wm_progress_lr'],
            coef=config['wm_progress_coef'],
            class_weights=tuple(progress_class_weights.tolist()),
            num_classes=progress_num_classes,
        )
        print(
            'Initialized diagnostic latent progress head with '
            f'class counts {progress_counts.tolist()} and weights '
            f'{progress_class_weights.tolist()} using '
            f'{progress_sampling}.',
            flush=True,
        )
    if config.get('wm_potential_enabled', False):
        if world_model is None:
            raise ValueError('wm_potential_enabled requires wm_enabled=True')
        potential_values = np.asarray(
            train_dataset['wm_potentials']
        ).reshape(-1)
        potential_target_mean = float(potential_values.mean())
        potential_target_std = float(potential_values.std())
        if potential_target_std <= 0:
            raise ValueError('Potential labels have zero variance')
        potential_edges = np.quantile(
            potential_values,
            np.linspace(0.0, 1.0, config['wm_potential_num_bins'] + 1),
        )
        potential_internal_edges = np.unique(potential_edges[1:-1])
        potential_bin_ids = np.digitize(
            potential_values, potential_internal_edges
        )
        potential_train_bin_indices = [
            np.flatnonzero(potential_bin_ids == bin_index)
            for bin_index in range(len(potential_internal_edges) + 1)
            if np.any(potential_bin_ids == bin_index)
        ]
        potential_bin_counts = [
            len(indices) for indices in potential_train_bin_indices
        ]
        world_model_potential = LatentPotentialTrainState.create(
            seed=FLAGS.seed + 4,
            latent_dim=config['wm_latent_dim'],
            hidden_dims=config['wm_potential_hidden_dims'],
            learning_rate=config['wm_potential_lr'],
            coef=config['wm_potential_coef'],
            target_mean=potential_target_mean,
            target_std=potential_target_std,
        )
        print(
            'Initialized diagnostic latent potential head with '
            f'range [{potential_values.min():.6f}, '
            f'{potential_values.max():.6f}], mean '
            f'{potential_target_mean:.6f}, std '
            f'{potential_target_std:.6f}, and quantile-bin counts '
            f'{potential_bin_counts}.',
            flush=True,
        )
    if config.get('wm_delta_enabled', False):
        if world_model is None:
            raise ValueError('wm_delta_enabled requires wm_enabled=True')
        delta_train_class_indices, delta_values = get_delta_start_groups(
            train_dataset
        )
        delta_target_mean = float(delta_values.mean())
        delta_target_std = float(delta_values.std())
        if delta_target_std <= 0:
            raise ValueError('Potential delta labels have zero variance')
        example_action_chunk = np.stack(
            [example_batch['actions']] * FLAGS.horizon_length,
            axis=-2,
        )
        world_model_delta = LatentPotentialDeltaTrainState.create(
            seed=FLAGS.seed + 5,
            latent_dim=config['wm_latent_dim'],
            example_action_chunk=example_action_chunk,
            hidden_dims=config['wm_delta_hidden_dims'],
            learning_rate=config['wm_delta_lr'],
            coef=config['wm_delta_coef'],
            target_mean=delta_target_mean,
            target_std=delta_target_std,
            nontrivial_threshold=config[
                'wm_delta_nontrivial_threshold'
            ],
        )
        print(
            'Initialized diagnostic latent potential-delta head with '
            f'mean {delta_target_mean:.6f}, std '
            f'{delta_target_std:.6f}, and decrease/neutral/increase '
            f'counts {[len(group) for group in delta_train_class_indices]}.',
            flush=True,
        )

    value_validation_batch = None
    if world_model_value is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 10_000)
        value_validation_dataset = Dataset.create(**val_dataset)
        value_validation_batch = value_validation_dataset.sample_sequence(
            config['batch_size'],
            sequence_length=FLAGS.horizon_length,
            discount=discount,
        )
        np.random.set_state(numpy_rng_state)

    progress_validation_batch = None
    if world_model_progress is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 30_000)
        progress_validation_dataset = Dataset.create(**val_dataset)
        validation_rewards = np.asarray(
            progress_validation_dataset['rewards']
        )
        validation_labels = np.clip(
            np.rint(validation_rewards + 3.0).astype(int),
            0,
            config['wm_progress_num_classes'] - 1,
        )
        validation_indices = []
        for class_index in range(config['wm_progress_num_classes']):
            class_indices = np.flatnonzero(
                validation_labels == class_index
            )
            if len(class_indices) == 0:
                continue
            validation_indices.append(
                np.random.choice(
                    class_indices,
                    size=config['wm_progress_validation_per_class'],
                    replace=(
                        len(class_indices)
                        < config['wm_progress_validation_per_class']
                    ),
                )
            )
        if not validation_indices:
            raise ValueError('No progress validation labels were found')
        validation_indices = np.concatenate(validation_indices)
        np.random.shuffle(validation_indices)
        progress_validation_batch = progress_validation_dataset.sample(
            len(validation_indices), idxs=validation_indices
        )
        np.random.set_state(numpy_rng_state)

    potential_validation_batch = None
    if world_model_potential is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 40_000)
        potential_validation_dataset = Dataset.create(
            **add_cube_potential_labels(val_dataset)
        )
        validation_potentials = np.asarray(
            potential_validation_dataset['wm_potentials']
        ).reshape(-1)
        validation_edges = np.quantile(
            validation_potentials,
            np.linspace(0.0, 1.0, config['wm_potential_num_bins'] + 1),
        )
        validation_internal_edges = np.unique(validation_edges[1:-1])
        validation_bin_ids = np.digitize(
            validation_potentials, validation_internal_edges
        )
        potential_validation_indices = []
        for bin_index in range(len(validation_internal_edges) + 1):
            bin_indices = np.flatnonzero(
                validation_bin_ids == bin_index
            )
            if len(bin_indices) == 0:
                continue
            potential_validation_indices.append(
                np.random.choice(
                    bin_indices,
                    size=config['wm_potential_validation_per_bin'],
                    replace=(
                        len(bin_indices)
                        < config['wm_potential_validation_per_bin']
                    ),
                )
            )
        potential_validation_indices = np.concatenate(
            potential_validation_indices
        )
        np.random.shuffle(potential_validation_indices)
        potential_validation_batch = potential_validation_dataset.sample(
            len(potential_validation_indices),
            idxs=potential_validation_indices,
        )
        np.random.set_state(numpy_rng_state)

    delta_validation_batch = None
    if world_model_delta is not None:
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 50_000)
        delta_validation_dataset = Dataset.create(
            **add_cube_potential_labels(val_dataset)
        )
        delta_validation_groups, _ = get_delta_start_groups(
            delta_validation_dataset
        )
        delta_validation_starts = np.concatenate(
            [
                np.random.choice(
                    group,
                    size=config['wm_delta_validation_per_class'],
                    replace=(
                        len(group)
                        < config['wm_delta_validation_per_class']
                    ),
                )
                for group in delta_validation_groups
            ]
        )
        np.random.shuffle(delta_validation_starts)
        delta_validation_batch = make_delta_batch(
            delta_validation_dataset, delta_validation_starts
        )
        np.random.set_state(numpy_rng_state)

    if FLAGS.restore_file is not None:
        restored = restore_agent_with_file(
            agent,
            FLAGS.restore_file,
            world_model=world_model,
            world_model_value=world_model_value,
            world_model_progress=world_model_progress,
            world_model_potential=world_model_potential,
            world_model_delta=world_model_delta,
        )
        restored_states = iter(
            restored if isinstance(restored, tuple) else (restored,)
        )
        agent = next(restored_states)
        if world_model is not None:
            world_model = next(restored_states)
        if world_model_value is not None:
            world_model_value = next(restored_states)
        if world_model_progress is not None:
            world_model_progress = next(restored_states)
        if world_model_potential is not None:
            world_model_potential = next(restored_states)
        if world_model_delta is not None:
            world_model_delta = next(restored_states)
        print(
            f"Restored checkpoint from {FLAGS.restore_file} "
            f"at global step {FLAGS.restore_step}",
            flush=True,
        )

    # Setup logging.
    prefixes = ["eval", "env"]
    if FLAGS.candidate_diagnostic_only:
        prefixes.append("candidate_diagnostic")
    if FLAGS.offline_steps > 0:
        prefixes.append("offline_agent")
    if FLAGS.online_steps > 0:
        prefixes.append("online_agent")

    logger = LoggingHelper(
        csv_loggers={prefix: CsvLogger(os.path.join(FLAGS.save_dir, f"{prefix}.csv"))
                    for prefix in prefixes},
        wandb_logger=wandb,
    )

    if FLAGS.eval_only:
        if FLAGS.restore_file is None:
            raise ValueError("--eval_only requires --restore_file")

        if FLAGS.candidate_diagnostic_only:
            raise ValueError(
                "--eval_only and --candidate_diagnostic_only are mutually "
                "exclusive"
            )

        sample_actions_fn = None
        if FLAGS.wm_score_eval_enabled:
            if world_model is None or world_model_value is None:
                raise ValueError(
                    "--wm_score_eval_enabled requires wm_enabled=True and "
                    "wm_value_enabled=True"
                )
            if config["actor_type"] != "best-of-n":
                raise ValueError(
                    "--wm_score_eval_enabled requires actor_type=best-of-n"
                )
            if not config["action_chunking"]:
                raise ValueError(
                    "--wm_score_eval_enabled requires action_chunking=True"
                )
            if FLAGS.wm_score_lambda < 0:
                raise ValueError("--wm_score_lambda must be non-negative")
            sample_actions_fn = functools.partial(
                world_model_value.sample_actions,
                world_model=world_model,
                agent=agent,
                score_lambda=FLAGS.wm_score_lambda,
            )
            print(
                "Evaluation-only world-model scoring enabled with "
                f"lambda={FLAGS.wm_score_lambda}.",
                flush=True,
            )

        elif FLAGS.wm_delta_score_eval_enabled:
            if world_model is None or world_model_delta is None:
                raise ValueError(
                    "--wm_delta_score_eval_enabled requires wm_enabled=True "
                    "and wm_delta_enabled=True"
                )
            if config["actor_type"] != "best-of-n":
                raise ValueError(
                    "--wm_delta_score_eval_enabled requires "
                    "actor_type=best-of-n"
                )
            if not config["action_chunking"]:
                raise ValueError(
                    "--wm_delta_score_eval_enabled requires "
                    "action_chunking=True"
                )
            if FLAGS.wm_delta_score_lambda < 0:
                raise ValueError(
                    "--wm_delta_score_lambda must be non-negative"
                )
            sample_actions_fn = functools.partial(
                world_model_delta.sample_actions,
                world_model=world_model,
                agent=agent,
                score_lambda=FLAGS.wm_delta_score_lambda,
            )
            print(
                "Evaluation-only world-model delta scoring enabled with "
                f"lambda={FLAGS.wm_delta_score_lambda}.",
                flush=True,
            )

        eval_info, eval_trajs, renders = evaluate(
            agent=agent,
            env=eval_env,
            action_dim=example_batch["actions"].shape[-1],
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            sample_actions_fn=sample_actions_fn,
            eval_seed=FLAGS.eval_seed,
        )
        logger.log(eval_info, "eval", step=log_step)

        if FLAGS.eval_seed is not None:
            episode_logger = CsvLogger(
                os.path.join(FLAGS.save_dir, "eval_episodes.csv")
            )
            for episode_index, trajectory in enumerate(eval_trajs):
                final_info = flatten(trajectory["info"][-1])
                success = final_info.get("success")
                if success is None:
                    success_values = [
                        value
                        for key, value in final_info.items()
                        if key.endswith(".success")
                    ]
                    success = (
                        success_values[0]
                        if len(success_values) == 1
                        else np.nan
                    )
                episode_logger.log(
                    {
                        "episode_seed": FLAGS.eval_seed + episode_index,
                        "success": success,
                        "return": np.sum(trajectory["reward"]),
                        "length": len(trajectory["reward"]),
                    },
                    step=episode_index,
                )
            episode_logger.close()

        if renders:
            run.log(
                {
                    "eval/video": get_wandb_video(
                        renders,
                        n_cols=min(len(renders), 5),
                    )
                },
                step=log_step,
            )

        for csv_logger in logger.csv_loggers.values():
            csv_logger.close()

        wandb.finish()
        return

    if FLAGS.candidate_diagnostic_only:
        if FLAGS.restore_file is None:
            raise ValueError(
                "--candidate_diagnostic_only requires --restore_file"
            )
        if world_model_value is not None:
            candidate_head = world_model_value
            candidate_head_name = "latent value"
        elif world_model_progress is not None:
            candidate_head = world_model_progress
            candidate_head_name = "latent progress"
        elif world_model_potential is not None:
            candidate_head = world_model_potential
            candidate_head_name = "latent potential"
        else:
            candidate_head = world_model_delta
            candidate_head_name = "latent potential delta"
        if world_model is None or candidate_head is None:
            raise ValueError(
                "--candidate_diagnostic_only requires wm_enabled=True and "
                "one diagnostic latent head enabled"
            )
        if config["actor_type"] != "best-of-n":
            raise ValueError(
                "--candidate_diagnostic_only requires actor_type=best-of-n"
            )
        if not config["action_chunking"]:
            raise ValueError(
                "--candidate_diagnostic_only requires action_chunking=True"
            )
        if FLAGS.candidate_diagnostic_batches <= 0:
            raise ValueError("--candidate_diagnostic_batches must be positive")

        diagnostic_dataset = Dataset.create(
            **(
                add_cube_potential_labels(val_dataset)
                if candidate_head_name == "latent potential"
                else val_dataset
            )
        )
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 20_000)
        diagnostic_rng = jax.random.PRNGKey(FLAGS.seed + 20_000)
        candidate_infos = []
        future_info = None
        try:
            if candidate_head_name == "latent progress":
                horizon_length = config["horizon_length"]
                max_start = diagnostic_dataset.size - horizon_length
                terminals = np.asarray(
                    diagnostic_dataset["terminals"]
                ).reshape(-1)
                terminal_prefix = np.concatenate(
                    ([0], np.cumsum(terminals > 0))
                )
                terminal_counts = (
                    terminal_prefix[
                        horizon_length:horizon_length + max_start
                    ]
                    - terminal_prefix[:max_start]
                )
                valid_starts = np.flatnonzero(terminal_counts == 0)
                future_rewards = np.asarray(
                    diagnostic_dataset["rewards"]
                ).reshape(-1)[valid_starts + horizon_length]
                future_labels = np.clip(
                    np.rint(future_rewards + 3.0).astype(int),
                    0,
                    config["wm_progress_num_classes"] - 1,
                )
                future_starts = []
                for class_index in range(
                    config["wm_progress_num_classes"]
                ):
                    class_starts = valid_starts[
                        future_labels == class_index
                    ]
                    if len(class_starts) == 0:
                        continue
                    future_starts.append(
                        np.random.choice(
                            class_starts,
                            size=config[
                                "wm_progress_validation_per_class"
                            ],
                            replace=(
                                len(class_starts)
                                < config[
                                    "wm_progress_validation_per_class"
                                ]
                            ),
                        )
                    )
                if not future_starts:
                    raise ValueError(
                        "No terminal-valid future progress samples found"
                    )
                future_starts = np.concatenate(future_starts)
                np.random.shuffle(future_starts)
                action_indices = (
                    future_starts[:, None]
                    + np.arange(horizon_length)[None, :]
                )
                future_batch = {
                    "observations": diagnostic_dataset["observations"][
                        future_starts
                    ],
                    "actions": diagnostic_dataset["actions"][
                        action_indices
                    ],
                    "target_observations": diagnostic_dataset[
                        "next_observations"
                    ][future_starts + horizon_length - 1],
                    "target_rewards": diagnostic_dataset["rewards"][
                        future_starts + horizon_length
                    ],
                }
                future_info = candidate_head.evaluate_predicted_future(
                    future_batch, world_model
                )
            elif candidate_head_name == "latent potential":
                horizon_length = config["horizon_length"]
                max_start = diagnostic_dataset.size - horizon_length
                terminals = np.asarray(
                    diagnostic_dataset["terminals"]
                ).reshape(-1)
                terminal_prefix = np.concatenate(
                    ([0], np.cumsum(terminals > 0))
                )
                terminal_counts = (
                    terminal_prefix[
                        horizon_length:horizon_length + max_start
                    ]
                    - terminal_prefix[:max_start]
                )
                valid_starts = np.flatnonzero(terminal_counts == 0)
                future_potentials = np.asarray(
                    diagnostic_dataset["wm_potentials"]
                ).reshape(-1)[valid_starts + horizon_length]
                future_edges = np.quantile(
                    future_potentials,
                    np.linspace(
                        0.0, 1.0, config["wm_potential_num_bins"] + 1
                    ),
                )
                future_internal_edges = np.unique(future_edges[1:-1])
                future_bin_ids = np.digitize(
                    future_potentials, future_internal_edges
                )
                future_starts_by_bin = []
                for bin_index in range(len(future_internal_edges) + 1):
                    bin_starts = valid_starts[
                        future_bin_ids == bin_index
                    ]
                    if len(bin_starts) == 0:
                        continue
                    future_starts_by_bin.append(
                        np.random.choice(
                            bin_starts,
                            size=config[
                                "wm_potential_validation_per_bin"
                            ],
                            replace=(
                                len(bin_starts)
                                < config[
                                    "wm_potential_validation_per_bin"
                                ]
                            ),
                        )
                    )
                if not future_starts_by_bin:
                    raise ValueError(
                        "No terminal-valid future potential samples found"
                    )
                future_starts = np.concatenate(future_starts_by_bin)
                np.random.shuffle(future_starts)
                action_indices = (
                    future_starts[:, None]
                    + np.arange(horizon_length)[None, :]
                )
                future_batch = {
                    "observations": diagnostic_dataset["observations"][
                        future_starts
                    ],
                    "actions": diagnostic_dataset["actions"][
                        action_indices
                    ],
                    "target_observations": diagnostic_dataset[
                        "next_observations"
                    ][future_starts + horizon_length - 1],
                    "target_potentials": diagnostic_dataset[
                        "wm_potentials"
                    ][future_starts + horizon_length],
                    "current_potentials": diagnostic_dataset[
                        "wm_potentials"
                    ][future_starts],
                }
                future_info = candidate_head.evaluate_predicted_future(
                    future_batch, world_model
                )
            elif candidate_head_name == "latent potential delta":
                future_info = candidate_head.evaluate(
                    delta_validation_batch, world_model
                )

            for _ in range(FLAGS.candidate_diagnostic_batches):
                diagnostic_batch = diagnostic_dataset.sample(
                    config["batch_size"]
                )
                diagnostic_rng, candidate_rng = jax.random.split(
                    diagnostic_rng
                )
                candidate_infos.append(
                    candidate_head.evaluate_candidates(
                        diagnostic_batch["observations"],
                        world_model,
                        agent,
                        candidate_rng,
                    )
                )
        finally:
            np.random.set_state(numpy_rng_state)

        candidate_info = jax.tree_util.tree_map(
            lambda *values: np.mean(np.asarray(values), axis=0),
            *candidate_infos,
        )
        candidate_batch_std = jax.tree_util.tree_map(
            lambda *values: np.std(np.asarray(values), axis=0),
            *candidate_infos,
        )
        if future_info is not None:
            candidate_info.update(
                {
                    f"future_{key}": np.asarray(value)
                    for key, value in future_info.items()
                }
            )
        batch_std_keys = [
            key
            for key in (
                "spearman_correlation",
                "score_correlation",
                "top1_agreement",
                "topk_overlap",
                "critic_choice_value_percentile",
                "value_choice_critic_percentile",
                "critic_choice_progress_percentile",
                "progress_choice_critic_percentile",
                "critic_choice_potential_percentile",
                "potential_choice_critic_percentile",
                "critic_choice_delta_percentile",
                "delta_choice_critic_percentile",
            )
            if key in candidate_info
        ]
        batch_std_keys.extend(
            key
            for key in candidate_info
            if key.startswith("lambda_")
            and key.rsplit("/", 1)[-1]
            in ("change_rate", "critic_percentile", "value_percentile")
        )
        for key in batch_std_keys:
            candidate_info[f"{key}_batch_std"] = candidate_batch_std[key]

        print(
            f"Read-only {candidate_head_name} candidate ranking diagnostic:",
            flush=True,
        )
        for key in sorted(candidate_info):
            print(
                f"  {key}: {float(candidate_info[key]):.6f}",
                flush=True,
            )
        logger.log(
            dict(candidate_info),
            "candidate_diagnostic",
            step=log_step,
        )

        for csv_logger in logger.csv_loggers.values():
            csv_logger.close()
        wandb.finish()
        return

    offline_init_time = time.time()
    # Offline RL
    for i in tqdm.tqdm(range(1, FLAGS.offline_steps + 1)):
        log_step += 1

        if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
            dataset_idx = (dataset_idx + 1) % len(dataset_paths)
            print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
            train_dataset, val_dataset = make_ogbench_env_and_datasets(
                FLAGS.env_name,
                dataset_path=dataset_paths[dataset_idx],
                compact_dataset=False,
                dataset_only=True,
                cur_env=env,
                add_info=cube_aux_enabled,
            )
            train_dataset = process_train_dataset(train_dataset)
            if FLAGS.wm_progress_only:
                replacement_labels = np.clip(
                    np.rint(
                        np.asarray(train_dataset['rewards']) + 3.0
                    ).astype(int),
                    0,
                    config['wm_progress_num_classes'] - 1,
                )
                progress_train_class_indices = [
                    np.flatnonzero(replacement_labels == class_index)
                    for class_index in range(
                        config['wm_progress_num_classes']
                    )
                    if np.any(replacement_labels == class_index)
                ]
            if FLAGS.wm_potential_only:
                replacement_potentials = np.asarray(
                    train_dataset['wm_potentials']
                ).reshape(-1)
                replacement_edges = np.quantile(
                    replacement_potentials,
                    np.linspace(
                        0.0, 1.0, config['wm_potential_num_bins'] + 1
                    ),
                )
                replacement_internal_edges = np.unique(
                    replacement_edges[1:-1]
                )
                replacement_bin_ids = np.digitize(
                    replacement_potentials, replacement_internal_edges
                )
                potential_train_bin_indices = [
                    np.flatnonzero(replacement_bin_ids == bin_index)
                    for bin_index in range(
                        len(replacement_internal_edges) + 1
                    )
                    if np.any(replacement_bin_ids == bin_index)
                ]
            if FLAGS.wm_delta_only:
                delta_train_class_indices, _ = get_delta_start_groups(
                    train_dataset
                )

        if FLAGS.wm_progress_only:
            num_present_classes = len(progress_train_class_indices)
            base_size, remainder = divmod(
                config['batch_size'], num_present_classes
            )
            progress_batch_indices = []
            for position, class_indices in enumerate(
                progress_train_class_indices
            ):
                class_batch_size = base_size + (position < remainder)
                progress_batch_indices.append(
                    np.random.choice(
                        class_indices,
                        size=class_batch_size,
                        replace=len(class_indices) < class_batch_size,
                    )
                )
            progress_batch_indices = np.concatenate(
                progress_batch_indices
            )
            np.random.shuffle(progress_batch_indices)
            batch = train_dataset.sample(
                len(progress_batch_indices),
                idxs=progress_batch_indices,
            )
        elif FLAGS.wm_potential_only:
            num_potential_bins = len(potential_train_bin_indices)
            base_size, remainder = divmod(
                config['batch_size'], num_potential_bins
            )
            potential_batch_indices = []
            for position, bin_indices in enumerate(
                potential_train_bin_indices
            ):
                bin_batch_size = base_size + (position < remainder)
                potential_batch_indices.append(
                    np.random.choice(
                        bin_indices,
                        size=bin_batch_size,
                        replace=len(bin_indices) < bin_batch_size,
                    )
                )
            potential_batch_indices = np.concatenate(
                potential_batch_indices
            )
            np.random.shuffle(potential_batch_indices)
            batch = train_dataset.sample(
                len(potential_batch_indices),
                idxs=potential_batch_indices,
            )
        elif FLAGS.wm_delta_only:
            num_delta_classes = len(delta_train_class_indices)
            base_size, remainder = divmod(
                config['batch_size'], num_delta_classes
            )
            delta_batch_starts = []
            for position, class_starts in enumerate(
                delta_train_class_indices
            ):
                class_batch_size = base_size + (position < remainder)
                delta_batch_starts.append(
                    np.random.choice(
                        class_starts,
                        size=class_batch_size,
                        replace=len(class_starts) < class_batch_size,
                    )
                )
            delta_batch_starts = np.concatenate(delta_batch_starts)
            np.random.shuffle(delta_batch_starts)
            batch = make_delta_batch(
                train_dataset, delta_batch_starts
            )
        else:
            batch = train_dataset.sample_sequence(
                config['batch_size'],
                sequence_length=FLAGS.horizon_length,
                discount=discount,
            )

        if (
            FLAGS.wm_progress_only
            or FLAGS.wm_potential_only
            or FLAGS.wm_delta_only
        ):
            offline_info = {}
        else:
            agent, offline_info = agent.update(batch)
            if world_model is not None:
                world_model, world_model_info = world_model.update(batch)
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model/{key}': value
                        for key, value in world_model_info.items()
                    },
                }
            if world_model_value is not None:
                world_model_value, value_info = world_model_value.update(
                    batch,
                    world_model,
                    agent,
                )
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model_value/{key}': value
                        for key, value in value_info.items()
                    },
                }
        if (
            world_model_progress is not None
            and not FLAGS.wm_potential_only
            and not FLAGS.wm_delta_only
        ):
            world_model_progress, progress_info = (
                world_model_progress.update(batch, world_model)
            )
            offline_info = {
                **offline_info,
                **{
                    f'world_model_progress/{key}': value
                    for key, value in progress_info.items()
                },
            }
        if (
            world_model_potential is not None
            and not FLAGS.wm_progress_only
            and not FLAGS.wm_delta_only
        ):
            world_model_potential, potential_info = (
                world_model_potential.update(batch, world_model)
            )
            offline_info = {
                **offline_info,
                **{
                    f'world_model_potential/{key}': value
                    for key, value in potential_info.items()
                },
            }
        if world_model_delta is not None:
            world_model_delta, delta_info = world_model_delta.update(
                batch, world_model
            )
            offline_info = {
                **offline_info,
                **{
                    f'world_model_delta/{key}': value
                    for key, value in delta_info.items()
                },
            }

        if i % FLAGS.log_interval == 0:
            if world_model_value is not None:
                heldout_info = world_model_value.evaluate(
                    value_validation_batch,
                    world_model,
                    agent,
                )
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model_value/heldout_{key}': value
                        for key, value in heldout_info.items()
                    },
                }
            if world_model_progress is not None:
                progress_heldout_info = world_model_progress.evaluate(
                    progress_validation_batch, world_model
                )
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model_progress/heldout_{key}': value
                        for key, value in progress_heldout_info.items()
                    },
                }
            if world_model_potential is not None:
                potential_heldout_info = world_model_potential.evaluate(
                    potential_validation_batch, world_model
                )
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model_potential/heldout_{key}': value
                        for key, value in potential_heldout_info.items()
                    },
                }
            if world_model_delta is not None:
                delta_heldout_info = world_model_delta.evaluate(
                    delta_validation_batch, world_model
                )
                offline_info = {
                    **offline_info,
                    **{
                        f'world_model_delta/heldout_{key}': value
                        for key, value in delta_heldout_info.items()
                    },
                }
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(
                agent,
                FLAGS.save_dir,
                log_step,
                world_model=world_model,
                world_model_value=world_model_value,
                world_model_progress=world_model_progress,
                world_model_potential=world_model_potential,
                world_model_delta=world_model_delta,
            )

        # eval
        if i == FLAGS.offline_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            # during eval, the action chunk is executed fully
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=example_batch["actions"].shape[-1],
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

            if renders:
                run.log(
                    {
                        "eval/video": get_wandb_video(
                            renders,
                            n_cols=min(len(renders), 5),
                        )
                    },
                    step=log_step,
                )

    # transition from offline to online
    replay_buffer = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
        
    ob, _ = env.reset()
    
    action_queue = []
    action_dim = example_batch["actions"].shape[-1]

    # Online RL
    update_info = {}

    from collections import defaultdict
    data = defaultdict(list)
    online_init_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.online_steps + 1)):
        log_step += 1
        online_rng, key = jax.random.split(online_rng)
        
        # during online rl, the action chunk is executed fully
        if len(action_queue) == 0:
            action = agent.sample_actions(observations=ob, rng=key)

            action_chunk = np.array(action).reshape(-1, action_dim)
            for action in action_chunk:
                action_queue.append(action)
        action = action_queue.pop(0)
        
        next_ob, int_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if FLAGS.save_all_online_states:
            state = env.get_state()
            data["steps"].append(i)
            data["obs"].append(np.copy(next_ob))
            data["qpos"].append(np.copy(state["qpos"]))
            data["qvel"].append(np.copy(state["qvel"]))
            if "button_states" in state:
                data["button_states"].append(np.copy(state["button_states"]))
        
        # logging useful metrics from info dict
        env_info = {}
        for key, value in info.items():
            if key.startswith("distance"):
                env_info[key] = value
        # always log this at every step
        logger.log(env_info, "env", step=log_step)

        if 'antmaze' in FLAGS.env_name and (
            'diverse' in FLAGS.env_name or 'play' in FLAGS.env_name or 'umaze' in FLAGS.env_name
        ):
            # Adjust reward for D4RL antmaze.
            int_reward = int_reward - 1.0
        elif is_robomimic_env(FLAGS.env_name):
            # Adjust online (0, 1) reward for robomimic
            int_reward = int_reward - 1.0

        if FLAGS.sparse:
            assert int_reward <= 0.0
            int_reward = (int_reward != 0.0) * -1.0

        transition = dict(
            observations=ob,
            actions=action,
            rewards=int_reward,
            terminals=float(done),
            masks=1.0 - terminated,
            next_observations=next_ob,
        )
        replay_buffer.add_transition(transition)
        
        # done
        if done:
            ob, _ = env.reset()
            action_queue = []  # reset the action queue
        else:
            ob = next_ob

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample_sequence(config['batch_size'] * FLAGS.utd_ratio, 
                        sequence_length=FLAGS.horizon_length, discount=discount)
            batch = jax.tree.map(lambda x: x.reshape((
                FLAGS.utd_ratio, config["batch_size"]) + x.shape[1:]), batch)

            agent, agent_info = agent.batch_update(batch)
            if world_model is not None:
                world_model, world_model_info = world_model.batch_update(batch)
                agent_info = {
                    **agent_info,
                    **{
                        f'world_model/{key}': value
                        for key, value in world_model_info.items()
                    },
                }
            if world_model_value is not None:
                world_model_value, value_info = world_model_value.batch_update(
                    batch,
                    world_model,
                    agent,
                )
                agent_info = {
                    **agent_info,
                    **{
                        f'world_model_value/{key}': value
                        for key, value in value_info.items()
                    },
                }
            if world_model_progress is not None:
                world_model_progress, progress_info = (
                    world_model_progress.batch_update(batch, world_model)
                )
                agent_info = {
                    **agent_info,
                    **{
                        f'world_model_progress/{key}': value
                        for key, value in progress_info.items()
                    },
                }
            update_info["online_agent"] = agent_info
            
        if i % FLAGS.log_interval == 0:
            if (
                world_model_value is not None
                and "online_agent" in update_info
            ):
                heldout_info = world_model_value.evaluate(
                    value_validation_batch,
                    world_model,
                    agent,
                )
                update_info["online_agent"] = {
                    **update_info["online_agent"],
                    **{
                        f'world_model_value/heldout_{key}': value
                        for key, value in heldout_info.items()
                    },
                }
            if (
                world_model_progress is not None
                and "online_agent" in update_info
            ):
                progress_heldout_info = world_model_progress.evaluate(
                    progress_validation_batch, world_model
                )
                update_info["online_agent"] = {
                    **update_info["online_agent"],
                    **{
                        f'world_model_progress/heldout_{key}': value
                        for key, value in progress_heldout_info.items()
                    },
                }
            for key, info in update_info.items():
                logger.log(info, key, step=log_step)
            update_info = {}

        if i == FLAGS.online_steps - 1 or \
            (FLAGS.eval_interval != 0 and i % FLAGS.eval_interval == 0):
            eval_info, _, renders = evaluate(
                agent=agent,
                env=eval_env,
                action_dim=action_dim,
                num_eval_episodes=FLAGS.eval_episodes,
                num_video_episodes=FLAGS.video_episodes,
                video_frame_skip=FLAGS.video_frame_skip,
            )
            logger.log(eval_info, "eval", step=log_step)

            if renders:
                run.log(
                    {
                        "eval/video": get_wandb_video(
                            renders,
                            n_cols=min(len(renders), 5),
                        )
                    },
                    step=log_step,
                )

        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(
                agent,
                FLAGS.save_dir,
                log_step,
                world_model=world_model,
                world_model_value=world_model_value,
                world_model_progress=world_model_progress,
                world_model_potential=world_model_potential,
                world_model_delta=world_model_delta,
            )

    end_time = time.time()

    for key, csv_logger in logger.csv_loggers.items():
        csv_logger.close()

    if FLAGS.save_all_online_states:
        c_data = {"steps": np.array(data["steps"]),
                 "qpos": np.stack(data["qpos"], axis=0), 
                 "qvel": np.stack(data["qvel"], axis=0), 
                 "obs": np.stack(data["obs"], axis=0), 
                 "offline_time": online_init_time - offline_init_time,
                 "online_time": end_time - online_init_time,
        }
        if len(data["button_states"]) != 0:
            c_data["button_states"] = np.stack(data["button_states"], axis=0)
        np.savez(os.path.join(FLAGS.save_dir, "data.npz"), **c_data)

    with open(os.path.join(FLAGS.save_dir, 'token.tk'), 'w') as f:
        f.write(run.url)

if __name__ == '__main__':
    app.run(main)
