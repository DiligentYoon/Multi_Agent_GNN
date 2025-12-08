import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
if sys.platform == "win32":
    try:
        import contextlib
        import diffcp.cone_program as dc
        def _dummy_threadpool_limits(*args, **kwargs):
            return contextlib.nullcontext()
        dc.threadpool_limits = _dummy_threadpool_limits
    except Exception:
        pass

import argparse
import yaml
import time
import torch
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import ray

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from task.env.nav_env import NavEnv
from task.worker.rolloutworker import RolloutWorker
from task.agent.ppo import PPOAgent
from task.model.models import RL_ActorCritic
from test_main_driver import viz_simulation_test
import gymnasium as gym

def main(cfg: dict, args: argparse.Namespace):
    # --- Setup ---
    torch.manual_seed(cfg['env']['seed'])
    np.random.seed(cfg['env']['seed'])
    device = torch.device(cfg['env']['device'])
    
    # --- Output Directory & Writer ---
    start_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("results", f"{start_time}_{cfg['agent']['experiment']['directory']}")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Results will be saved to {experiment_dir}")
    print(f"Checkpoints will be saved to {checkpoint_dir}")
    writer = SummaryWriter(log_dir=experiment_dir)

    # --- Initialize Ray ---
    ray.init()
    print("Ray initialized.")

    # --- Create Central Learner Agent ---
    # Dummy values for H, W - the model architecture doesn't strictly depend on them
    eval_env = NavEnv(episode_index=0, device=device, cfg=cfg['env'], is_train=False)
    pr = eval_env.cfg.pooling_downsampling_rate
    num_agents = cfg['env']['num_agent']
    observation_space = gym.spaces.Box(0, 1, (8 + num_agents, 
                                              eval_env.obs_manager.global_map_size // pr, 
                                              eval_env.obs_manager.global_map_size // pr), dtype='uint8')
    action_space = gym.spaces.Box(0, (eval_env.obs_manager.global_map_size // pr) * (eval_env.obs_manager.global_map_size // pr) - 1, (num_agents,), dtype='int32')
    del eval_env
    
    model_cfg = cfg['model']
    for key, value in model_cfg.items():
        if key in ['actor_lr', 'critic_lr', 'eps'] and isinstance(value, str):
            model_cfg[key] = float(value)
    learner_model = RL_ActorCritic(observation_space.shape, action_space,
                                   model_type=model_cfg['model_type'],
                                   base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                                'use_history': model_cfg['use_history'],
                                                'ablation': model_cfg['ablation']},
                                   lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                   eps=model_cfg['eps']).to(device)
    learner_agent = PPOAgent(learner_model, device, cfg['agent'])
    print("Central learner agent created.")

    if args.checkpoint is not None:
        file_path = args.checkpoint
        learner_agent.model.load(file_path, device)
        print(f"Loaded checkpoint from {file_path}")

    # --- Create Rollout Workers ---
    num_workers = cfg['ray']['num_workers']
    workers = [RolloutWorker.remote(worker_id=i, cfg=cfg) for i in range(num_workers)]
    print(f"{num_workers} rollout workers created.")

    # --- Training Loop ---
    total_timesteps = cfg['train']['timesteps']
    rollout = num_workers * cfg['agent']['buffer']['rollout']
    
    
    per_step_reward = deque(maxlen=100)
    rollout_reward = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    action_losses = deque(maxlen=100)
    dist_entropies = deque(maxlen=100)
    
    global_step = 0
    iteration = 0

    print("=== Starting Distributed Training ===")
    while global_step < total_timesteps:
        iteration += 1
        
        # ============== Broadcasting the Parameters ==============
        current_weights = learner_agent.model.network.state_dict()
        cpu_weights = {k: v.to('cpu') for k, v in current_weights.items()}
        weights_ref = ray.put(cpu_weights)
        set_weight_futures = [worker.set_weights.remote(weights_ref) for worker in workers]
        ray.get(set_weight_futures) # Wait for all workers to update

        # ============== Parallel Rollouts =================
        t1_rollout = time.time()
        rollout_futures = [worker.sample.remote() for worker in workers]
        results = ray.get(rollout_futures)
        rollout_buffers, rollout_infos = zip(*results)
        t2_rollout = time.time()

        # Perform learning updates using the collected data
        iter_v_loss, iter_a_loss, iter_d_entropy = 0, 0, 0
        iter_per_step_reward, iter_rollout_reward = 0, 0
        
        # Accumulators for true global metrics
        total_episodes_completed = 0
        total_successes = 0
        total_coverage = 0.0
        total_avg_episode_steps = 0.0

        t1 = time.time()
        for buffer, rollout_info in zip(rollout_buffers, rollout_infos):
            # The buffer from the worker already has returns computed.
            value_loss, action_loss, dist_entropy = learner_agent.update(buffer.to(device))
            
            # Aggregate raw counts from each worker
            total_successes += rollout_info.get('total_successes', 0)
            total_episodes_completed += rollout_info.get('episodes_completed', 0)
            total_coverage += rollout_info.get('total_coverage', 0)
            total_avg_episode_steps += rollout_info.get('episode_step', 0)

            iter_per_step_reward += torch.sum(buffer.rewards).item() / rollout
            iter_rollout_reward += torch.sum(buffer.rewards).item()
            if value_loss > 0: # Assuming positive loss indicates a valid update
                iter_v_loss += value_loss
                iter_a_loss += action_loss
                iter_d_entropy += dist_entropy
        t2 = time.time()

        # Aggregate and log losses
        num_updates = len(rollout_buffers)
        if num_updates > 0:
            per_step_reward.append(iter_per_step_reward / num_updates)
            rollout_reward.append(iter_rollout_reward / num_updates)
            value_losses.append(iter_v_loss / num_updates)
            action_losses.append(iter_a_loss / num_updates)
            dist_entropies.append(iter_d_entropy / num_updates)

        global_step += rollout

        # Tensorboard Logging and Checkpointing
        if iteration % cfg['agent']['experiment']['write_interval'] == 0 and len(value_losses) > 0:
            mean_per_step_reward = np.mean(per_step_reward)
            mean_rollout_reward = np.mean(rollout_reward)
            mean_v_loss = np.mean(value_losses)
            mean_a_loss = np.mean(action_losses)
            mean_d_entropy = np.mean(dist_entropies)
            
            writer.add_scalar('Reward/Per_step', mean_per_step_reward, global_step)
            writer.add_scalar('Reward/Rollout', mean_rollout_reward, global_step)
            writer.add_scalar('Loss/Value', mean_v_loss, global_step)
            writer.add_scalar('Loss/Action', mean_a_loss, global_step)
            writer.add_scalar('Loss/Entropy', mean_d_entropy, global_step)

        if iteration % cfg['agent']['experiment']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_{global_step}.pt")
            learner_agent.model.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Evaluation and Visualization
        if iteration % cfg['train']['eval_freq'] == 0 or iteration == 1:
            eval_dir = os.path.join(experiment_dir, "eval")
            os.makedirs(eval_dir, exist_ok=True)
            gif_path_eval = os.path.join(eval_dir, f"eval_iter_{iteration}.gif")
            print(f"--- Running Evaluation & Visualization at Iteration {iteration} ---")
            t_r, c_r = viz_simulation_test(cfg, steps=20, is_train=False, gif_path=gif_path_eval, agent_model=learner_agent.model)
            print(f"Total Reward / Coverage Rate : {t_r:.2f} / {c_r:.2f}")
        
        # Calculate true global averages for logging
        num_workers = num_updates
        global_success_rate = (total_successes / total_episodes_completed) if total_episodes_completed > 0 else 0
        global_coverage_rate = (total_coverage / total_episodes_completed) if total_episodes_completed > 0 else 0
        global_avg_episode_steps = total_avg_episode_steps / num_workers if num_workers > 0 else 0

        # CLI Logging about the training process
        content_width = 64
        line_header = f"Training Iteration {iteration} Report"
        line_rollout_time = f"Rollout Time      : {t2_rollout - t1_rollout:6.2f} sec"
        line_train_time = f"Training Time     : {t2 - t1:6.2f} sec"
        line_episode_step = f"Avg Episode Step  : {global_avg_episode_steps:6.2f} steps"
        line_episode_success = f"Avg Success Rate  : {100 * global_success_rate:6.2f} %"
        line_per_step_reward = f"Per-Step Rewards  : {iter_per_step_reward / num_updates:6.2f}"
        line_rollout_reward = f"Rollout Rewards   : {iter_rollout_reward / num_updates:6.2f}"
        line_value_loss = f"Value Loss        : {iter_v_loss / num_updates:6.2f}"
        line_policy_loss = f"Policy Loss       : {iter_a_loss / num_updates:6.2f}"
        
        print(f" ________________________________________________________________")
        print(f"|                                                                |")
        print(f"|{line_header.center(content_width)}|")
        print(f"|________________________________________________________________|")
        print(f"|                                                                |")
        print(f"| {line_rollout_time:<{content_width-1}}|")
        print(f"| {line_train_time:<{content_width-1}}|")
        print(f"| {line_episode_step:<{content_width-1}}|")
        print(f"| {line_episode_success:<{content_width-1}}|")
        print(f"| {line_per_step_reward:<{content_width-1}}|")
        print(f"| {line_rollout_reward:<{content_width-1}}|")
        print(f"| {line_value_loss:<{content_width-1}}|")
        print(f"| {line_policy_loss:<{content_width-1}}|")
        print(f"|________________________________________________________________|")


        # CLI Logging about the environment information
        line_header_env = f"Environment Information Report"
        line_converage_rate = f"Coverage Rate   : {100 * global_coverage_rate:6.2f} %"
        print(f"|                                                                |")
        print(f"|{line_header_env.center(content_width)}|")
        print(f"|________________________________________________________________|")
        print(f"|                                                                |")
        print(f"| {line_converage_rate:<{content_width-1}}|")
        print(f"|________________________________________________________________|")


    # --- Cleanup ---
    writer.close()
    ray.shutdown()
    print("=== Distributed Training Finished ===")

if __name__ == '__main__':
    # --- Load Config ---
    with open("config/nav_ppo_cfg.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

    args = parser.parse_args()
    main(cfg, args)