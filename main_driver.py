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
import gymnasium as gym

def main(cfg: dict):
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
    temp_env = NavEnv(episode_index=0, device=device, cfg=cfg['env'])
    pr = temp_env.cfg.pooling_downsampling_rate
    num_agents = cfg['env']['num_agent']
    temp_H, temp_W = temp_env.map_info.H, temp_env.map_info.W 
    observation_space = gym.spaces.Box(0, 1, (8 + num_agents, temp_H // pr, temp_W // pr), dtype='uint8')
    action_space = gym.spaces.Box(0, (temp_H // pr) * (temp_W // pr) - 1, (num_agents,), dtype='int32')
    del temp_env

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

    # --- Create Rollout Workers ---
    num_workers = cfg['ray']['num_workers']
    workers = [RolloutWorker.remote(worker_id=i, cfg=cfg) for i in range(num_workers)]
    print(f"{num_workers} rollout workers created.")

    # --- Training Loop ---
    total_timesteps = cfg['train']['timesteps']
    rollout = num_workers * cfg['agent']['buffer']['rollout']
    rollout_log_interval = 2
    
    per_step_reward = deque(maxlen=100)
    value_losses = deque(maxlen=100)
    action_losses = deque(maxlen=100)
    dist_entropies = deque(maxlen=100)
    
    global_step = 0
    iteration = 0

    print("=== Starting Distributed Training ===")
    while global_step < total_timesteps:
        iteration += 1
        
        # Broadcast the latest policy weights to all workers
        current_weights = learner_agent.model.network.state_dict()
        cpu_weights = {k: v.to('cpu') for k, v in current_weights.items()}
        weights_ref = ray.put(cpu_weights)
        set_weight_futures = [worker.set_weights.remote(weights_ref) for worker in workers]
        ray.get(set_weight_futures) # Wait for all workers to update

        # Trigger parallel rollouts
        rollout_futures = [worker.sample.remote() for worker in workers]
        
        # Gather the collected rollout buffers
        rollout_buffers = ray.get(rollout_futures)

        # Perform learning updates using the collected data
        t1 = time.time()
        iter_v_loss, iter_a_loss, iter_d_entropy, iter_reward = 0, 0, 0, 0
        print(f"======== Starting Training Iteration {iteration} ========")
        for buffer in rollout_buffers:
            # The buffer from the worker already has returns computed.
            value_loss, action_loss, dist_entropy = learner_agent.update(buffer.to(device))
            
            iter_reward += torch.sum(buffer.rewards).item() / rollout
            if value_loss > 0: # Assuming positive loss indicates a valid update
                iter_v_loss += value_loss
                iter_a_loss += action_loss
                iter_d_entropy += dist_entropy
        t2 = time.time()
        # Aggregate and log losses
        num_updates = len(rollout_buffers)
        if num_updates > 0:
            per_step_reward.append(iter_reward / num_updates)
            value_losses.append(iter_v_loss / num_updates)
            action_losses.append(iter_a_loss / num_updates)
            dist_entropies.append(iter_d_entropy / num_updates)

        global_step += rollout

        # Tensorboard Logging and Checkpointing
        if iteration % cfg['agent']['experiment']['write_interval'] == 0 and len(value_losses) > 0:
            mean_per_step_reward = np.mean(per_step_reward)
            mean_v_loss = np.mean(value_losses)
            mean_a_loss = np.mean(action_losses)
            mean_d_entropy = np.mean(dist_entropies)
            
            print(f"Global Step: {global_step}/{total_timesteps} | Iteration: {iteration}")
            print(f"  Rewards : {mean_per_step_reward:.4f}")
            print(f"  Losses -> Value: {mean_v_loss:.4f}, Action: {mean_a_loss:.4f}, Entropy: {mean_d_entropy:.4f}")
            
            writer.add_scalar('Reward/Per_step', mean_per_step_reward, global_step)
            writer.add_scalar('Loss/Value', mean_v_loss, global_step)
            writer.add_scalar('Loss/Action', mean_a_loss, global_step)
            writer.add_scalar('Loss/Entropy', mean_d_entropy, global_step)

        if iteration % cfg['agent']['experiment']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"agent_{global_step}.pt")
            learner_agent.model.save(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # CLI Logging
        print(f"Time for train : {t2 - t1:.2f} sec")
        print(f"Rewards : {iter_reward / num_updates:.2f}")
        print(f"Value Loss : {iter_v_loss / num_updates:.2f}")
        print(f"Policy Loss : {iter_a_loss / num_updates:.2f}")
        print(f"============= Learning Progress at Iteration {iteration} ==============")

    # --- Cleanup ---
    writer.close()
    ray.shutdown()
    print("=== Distributed Training Finished ===")

if __name__ == '__main__':
    # --- Load Config ---
    with open("config/nav_ppo_cfg.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    main(cfg)