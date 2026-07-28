import glob, tqdm, wandb, os, json, random, time, jax
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

from evaluation import evaluate
from agents import agents
from agents.world_model import LatentValueTrainState, WorldModelTrainState
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
    'Only compare critic and latent-value rankings on best-of-N candidates.',
)
flags.DEFINE_integer(
    'candidate_diagnostic_batches',
    16,
    'Number of held-out batches used by candidate_diagnostic_only.',
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
    def process_train_dataset(ds):
        """
        Process the train dataset to 
            - handle dataset proportion
            - handle sparse reward
            - convert to action chunked dataset
        """

        ds = Dataset.create(**ds)
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

    if FLAGS.restore_file is not None:
        restored = restore_agent_with_file(
            agent,
            FLAGS.restore_file,
            world_model=world_model,
            world_model_value=world_model_value,
        )
        if world_model is None:
            agent = restored
        elif world_model_value is None:
            agent, world_model = restored
        else:
            agent, world_model, world_model_value = restored
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

        for csv_logger in logger.csv_loggers.values():
            csv_logger.close()

        wandb.finish()
        return

    if FLAGS.candidate_diagnostic_only:
        if FLAGS.restore_file is None:
            raise ValueError(
                "--candidate_diagnostic_only requires --restore_file"
            )
        if world_model is None or world_model_value is None:
            raise ValueError(
                "--candidate_diagnostic_only requires wm_enabled=True and "
                "wm_value_enabled=True"
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

        diagnostic_dataset = Dataset.create(**val_dataset)
        numpy_rng_state = np.random.get_state()
        np.random.seed(FLAGS.seed + 20_000)
        diagnostic_rng = jax.random.PRNGKey(FLAGS.seed + 20_000)
        candidate_infos = []
        try:
            for _ in range(FLAGS.candidate_diagnostic_batches):
                diagnostic_batch = diagnostic_dataset.sample(
                    config["batch_size"]
                )
                diagnostic_rng, candidate_rng = jax.random.split(
                    diagnostic_rng
                )
                candidate_infos.append(
                    world_model_value.evaluate_candidates(
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
        for key in (
            "spearman_correlation",
            "score_correlation",
            "top1_agreement",
            "topk_overlap",
            "critic_choice_value_percentile",
            "value_choice_critic_percentile",
        ):
            candidate_info[f"{key}_batch_std"] = candidate_batch_std[key]

        print("Read-only candidate ranking diagnostic:", flush=True)
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
            )
            train_dataset = process_train_dataset(train_dataset)

        batch = train_dataset.sample_sequence(config['batch_size'], sequence_length=FLAGS.horizon_length, discount=discount)

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
            logger.log(offline_info, "offline_agent", step=log_step)
        
        # saving
        if FLAGS.save_interval > 0 and i % FLAGS.save_interval == 0:
            save_agent(
                agent,
                FLAGS.save_dir,
                log_step,
                world_model=world_model,
                world_model_value=world_model_value,
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
