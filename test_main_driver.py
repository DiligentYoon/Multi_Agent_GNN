import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OMP_NUM_THREADS"] = "1"
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym
import datetime
import copy
import argparse
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from typing import List
from task.env.nav_env import NavEnv
from task.agent.sac import SACAgent
from task.agent.ppo import Agent, PPOAgent
from task.model.models import RL_ActorCritic
from task.model.models_ver_2 import RL_Policy
from task.model.models_commaping import RL_CoMapping_Policy
from task.buffer.rolloutbuffer import RolloutBuffer, CoMappingRolloutBuffer

from visualization import draw_frame, plot_cbf_values

PATH = os.path.join(os.getcwd())
SPECIFIC_PATH = ""


def create_model(cfg: dict, observation_space: gym.Space, action_space: gym.Space, device: torch.device, model_version: int):
    """
    Helper function to create models based on the config.

        Inputs:
            cfg: Model Configuration Dictionary
            observation_space : observation with respect to specific env
            action_space : action with repsect to specific env
            device : ["cpu", "cuda"]
        
        Returns:
            actor_critic : Actor Critic Model [policy_net, critic_net]
    """
    model_cfg = cfg
    for key, value in model_cfg.items():
        if key in ['actor_lr', 'critic_lr', 'eps'] and isinstance(value, str):
            model_cfg[key] = float(value)
    if model_version == 1:
        actor_critic = RL_ActorCritic(observation_space.shape, action_space,
                                model_type=model_cfg['model_type'],
                                base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                            'use_history': model_cfg['use_history'],
                                            'ablation': model_cfg['ablation']},
                                lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                eps=model_cfg['eps']).to(device)
    elif model_version == 2:
        actor_critic = RL_Policy(observation_space.shape, action_space,
                                 model_type=model_cfg['model_type'],
                                 base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                            'use_history': model_cfg['use_history'],
                                            'ablation': model_cfg['ablation']},
                                 lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                 eps=model_cfg['eps']).to(device)
    else:
        actor_critic = RL_CoMapping_Policy(observation_space.shape, action_space,
                                            model_type=model_cfg['model_type'],
                                            base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                                         'use_history': model_cfg['use_history'],
                                                         'ablation': model_cfg['ablation']},
                                            lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                            eps=model_cfg['eps']).to(device)
    return actor_critic


def create_agent(cfg: dict, model, num_agents: int, eval_freq: int, 
                 observation_space: gym.Space, action_space: gym.Space, device: torch.device) -> tuple[Agent, RolloutBuffer]:
    """
    Helper function to create agent and corresponding buffer based on the config.

        Inputs:
            cfg: Agent Configuration Dictionary
            models: Actor Critic Model [policy_net, critic_net]
        
        Returns:
            agent : on policy (PPO) or off policy (SAC)
    """
    agent_cfg = cfg
    buffer_cfg = agent_cfg["buffer"]
    algorithm = agent_cfg.get("alg", "PPO")
    if algorithm == "PPO":
        buffer = CoMappingRolloutBuffer(buffer_cfg["rollout"], 
                                        num_envs=1, 
                                        eval_freq=eval_freq,
                                        num_repeats=agent_cfg["num_gae_block"],
                                        num_agents=1, # Centralized Network
                                        obs_shape=observation_space.shape,
                                        action_space=action_space,
                                        rec_state_size=1,
                                        extras_size=num_agents * 6
                                        ).to(device)
        agent = PPOAgent(model, device, agent_cfg)
    elif algorithm == "SAC":
        raise ValueError("Not Buffer Implementation yet.")
        agent = SACAgent()
    else:
        raise ValueError("Unvalid Algorithm Types.")

    return agent, buffer


def run_simulation_test(args: argparse.Namespace,
                        cfg: dict, 
                        steps: int, 
                        out_dir: str = 'test_results', 
                        visualize: bool = True, 
                        num_episodes: int = 10):
    """
    Runs a simulation test for multiple episodes, collects metrics, and saves them.
        Inputs:
            cfg: Total Configuration
            steps: maximum simulation steps per episode
            out_dir: visualization and results save directory
            visualize: If True, generates and saves a GIF for the first episode.
            num_episodes: The number of episodes to run for evaluation.
    """
    print(f"=== Starting Simulation Test for {num_episodes} Episodes ===")
    
    # --- Device ---
    device = torch.device(cfg['env']['device'])

    # --- Environment ---
    if args.map_type is not None:
        cfg['env']['map']['type'] = args.map_type
    else:
        cfg['env']['map']['type'] = 'corridor'
        print('[Warning] Map type is None. So, Corridor map is forced to choose.')
    
    # Use a deepcopy of cfg for env creation to avoid modification issues
    env_cfg = copy.deepcopy(cfg['env'])
    env = NavEnv(episode_index=0, device=device, cfg=env_cfg, is_train=False, max_episode_steps=steps)
    
    # --- Agent & Models ---
    pr = env.cfg.pooling_downsampling_rate
    num_agents = cfg['env']['num_agent']
    observation_space = gym.spaces.Box(0, 1, (8 + num_agents, 
                                              env.obs_manager.global_map_size // pr, 
                                              env.obs_manager.global_map_size // pr), dtype='uint8')
    action_space = gym.spaces.Box(0, (env.obs_manager.global_map_size // pr) * (env.obs_manager.global_map_size // pr) - 1, (num_agents,), dtype='int32')
    
    actor_critic_model = create_model(cfg['model'], observation_space, action_space, device, args.version)
    agent, buffer = create_agent(cfg['agent'], actor_critic_model, num_agents,
                                 cfg['train']['eval_freq'], observation_space, action_space, device)
    if args.checkpoint is not None:
        agent.load(args.checkpoint, device=device)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # --- Data Storage for All Episodes ---
    evaluation_results = []

    for episode_num in range(num_episodes):
        current_seed = (args.seed if args.seed != -1 else 0) + episode_num
        print(f"Starting Episode {episode_num + 1}/{num_episodes} with Seed {current_seed}")

        # Set global seeds for consistent environment generation per episode
        torch.manual_seed(current_seed)
        np.random.seed(current_seed)

        # --- Per-Episode Data Tracking ---
        episode_min_inter_agent_dist = float('inf')
        connected_steps = 0
        
        # --- Visualization Setup (only for the first episode if enabled) ---
        visualize_this_episode = visualize
        frames: List[np.ndarray] = []
        fig, ax1, ax2 = None, None, None
        if visualize_this_episode:
            print("Visualization is ON for this episode.")
            if args.map_type == 'corridor':
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        path_history  = [[] for _ in range(num_agents)]
        cbf_history   = [[] for _ in range(num_agents)]

        # --- Callback for rendering each physics step ---
        def render_callback(env_instance: NavEnv):
            if not visualize_this_episode: return
            for j in range(num_agents):
                path_history[j].append((env_instance.robot_locations[j, 0], env_instance.robot_locations[j, 1]))

            connectivity_pairs = []
            for i in range(env_instance.num_agent):
                pos1 = env_instance.robot_locations[i]
                if not env_instance.root_mask[i]:
                    parent_id = env_instance.connectivity_graph.get_parent(i)
                    pos2 = env_instance.robot_locations[parent_id]
                else:
                    pos2 = pos1
                connectivity_pairs.append((pos1, pos2))

            viz_data = {
                "paths": path_history, "obs_local": env_instance.obstacle_states,
                "connectivity_pairs": connectivity_pairs, "target_local": env_instance.cbf_infos["nominal"]["p_targets"],
                "connectivity_trajs": env_instance.connectivity_traj,
            }
            draw_frame(ax1, ax2, env_instance, viz_data)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf, dtype=np.uint8)[..., :3]
            frames.append(frame.copy())

        # ================ Simulation Start for One Episode ====================
        obs, _, info = env.reset(episode_index=current_seed)

        l = buffer.mini_step * buffer.mini_step_size
        h = (buffer.mini_step + 1) * buffer.mini_step_size
        buffer.obs[0][l:h].copy_(obs.view(1, *observation_space.shape))
        buffer.extras[0][l:h].copy_(info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate)
        ll, lh = l-buffer.mini_step_size, h-buffer.mini_step_size
        if lh == 0: lh = buffer.mini_step_size * buffer.num_rollout_blocks
        buffer.obs[-1][ll:lh].copy_(buffer.obs[0][l:h])
        buffer.rec_states[-1][ll:lh].copy_(buffer.rec_states[0][l:h])
        buffer.extras[-1][ll:lh].copy_(buffer.extras[0][l:h])

        rec_states = buffer.rec_states[0][l:h]
        mask = buffer.masks[0][l:h]
        
        final_step_num = 0
        for step_num in range(steps):
            final_step_num = step_num
            with torch.no_grad():
                values, actions, action_log_probs, rec_states, _ = agent.act(
                    obs.view(1, *observation_space.shape), rec_states, mask,
                    info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate,
                    deterministic=True
                )

            next_obs, _, reward, terminated, truncated, next_info = env.step(actions, on_physics_step=render_callback)
            done = torch.logical_or(terminated, truncated)

            buffer.insert(next_obs.view(1, *observation_space.shape), 
                          rec_states, actions, action_log_probs, 
                          values, reward, ~done, 
                          next_info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate)

            # --- Metrics Calculation (per step) ---
            # 1. Minimum Inter-Agent Distance
            locations = env.robot_locations
            diff = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dist_matrix, np.inf)
            current_min_dist = np.min(dist_matrix)
            episode_min_inter_agent_dist = min(episode_min_inter_agent_dist, current_min_dist)

            # 2. Connectivity Maintenance & Cbf History for Plotting
            is_connected_this_step = True
            for j in range(num_agents):
                # For CBF history plotting (if enabled)
                obs_state = env.obstacle_states[j, :env.num_obstacles[j]]
                obs_avoid_val = env.cfg.sensor_range**2 if env.num_obstacles[j] == 0 else np.min(np.linalg.norm(obs_state, axis=1))**2
                
                # For connectivity check
                p_c = env.cbf_infos["safety"]["p_c_agent"][j].reshape(-1)
                agent_conn_val = env.neighbor_radius**2 # Default to a high value if no parent
                if len(p_c) > 0:
                    conn_agent_dist_sq = p_c[0]**2 + p_c[1]**2
                    agent_conn_val = env.neighbor_radius**2 - conn_agent_dist_sq
                    if agent_conn_val < 0:
                        is_connected_this_step = False
                
                if visualize_this_episode:
                    cbf_history[j].append({"obs_avoid": obs_avoid_val - env.cfg.d_safe**2, "agent_conn": agent_conn_val})

            if is_connected_this_step:
                connected_steps += 1

            if num_episodes == 1:
                print(f"Episode {episode_num+1}, Step {step_num+1}/{steps}, Reward: {reward.item():.2f}")

            obs, info, mask = copy.deepcopy(next_obs), copy.deepcopy(next_info), copy.deepcopy(~done)
            
            if done:
                break
        
        print(f"Episode {episode_num+1} finished at step {final_step_num + 1}.")
        print(f"Results (Success/Failure) : {env.is_success}")

        # --- Metrics Calculation (end of episode) ---
        traversal_time_sec = (final_step_num + 1) * env.dt * env.decimation
        coverage_rate = env.prev_explored_region / (env.map_info.H * env.map_info.W)
        connectivity_rate = connected_steps / (final_step_num + 1)

        episode_results = {
            "episode_seed": current_seed,
            "success": bool(env.is_success),
            "traversal_time_sec": traversal_time_sec,
            "coverage_rate_percent": coverage_rate * 100,
            "min_inter_agent_dist_m": episode_min_inter_agent_dist,
            "connectivity_maintenance_rate_percent": connectivity_rate * 100
        }
        evaluation_results.append(episode_results)

        if visualize_this_episode:
            plt.close(fig)
            gif_path = os.path.join(out_dir, f'simulation_ep_{episode_num+1}_seed_{current_seed}.gif')
            print(f"Saving GIF to {gif_path}...")
            imageio.mimsave(gif_path, frames, fps=30)
            print("GIF saved.")
            plot_cbf_values(cbf_history, env.dt * env.decimation, num_agents, save_path=os.path.join(out_dir, f'cbf_values_ep_{episode_num+1}_seed_{current_seed}.png'))

    # --- Final Aggregation and Saving ---
    df = pd.DataFrame(evaluation_results)
    
    # Calculate overall success rate
    success_rate = df['success'].mean() * 100
    print("" + "="*50)
    print("=== Overall Evaluation Results ===")
    print(f"Success Rate: {success_rate:.2f}%")
    print("Per-Episode Metrics:")
    print(df)
    
    # Save to CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'evaluation_metrics_{args.map_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")
    print("="*50)


def viz_simulation_test(cfg: dict,
                        is_train: bool, 
                        steps: int,
                        gif_path: str = 'test_results', 
                        agent_model = None,
                        map_type: int = 0):
    """
    Runs a simulation test, generating a GIF and plots.
        Inputs:
            cfg: Total Configuration
            stpes: maximum simulation steps
            out_dir: visualization save directory
            visualize : If True, generates and saves a GIF and plots. 
                        If False, runs the simulation without visualization.
            load_file_path: Path to load checkpoint from (used if agent_model is None)
            agent_model: Pre-loaded model object (used for evaluation during training)
    """
    # Save current global RNG state to preserve training randomness
    rng_state_np = np.random.get_state()
    rng_state_torch = torch.get_rng_state()

    # Set deterministic seed for evaluation to ensure consistent environment conditions
    eval_seed = 25
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)

    try:
        cfg = copy.deepcopy(cfg)
        # --- Device ---
        device = torch.device(cfg['env']['device'])
        # --- Environment ---
        type_list = ['corridor', 'maze', 'random', 'single_maze']
        cfg['env']['map']['type'] = type_list[map_type]
        env = NavEnv(episode_index=0, device=device, cfg=cfg['env'], is_train=is_train, max_episode_steps=steps)
        # --- Agent & Models ---
        pr = env.cfg.pooling_downsampling_rate
        num_agents = cfg['env']['num_agent']
        observation_space = gym.spaces.Box(0, 1, (8 + num_agents, env.obs_manager.global_map_size // pr, env.obs_manager.global_map_size // pr), dtype='uint8')
        action_space = gym.spaces.Box(0, (env.obs_manager.global_map_size // pr) * (env.obs_manager.global_map_size // pr) - 1, (num_agents,), dtype='int32')
        
        actor_critic_model = agent_model
        agent, buffer = create_agent(cfg['agent'], actor_critic_model, num_agents,
                                     cfg['train']['eval_freq'], observation_space, action_space, device)
    
        # --- Visualization and Data Tracking Setup ---
        frames: List[np.ndarray] = []
        fig, ax1, ax2 = None, None, None
        if map_type == 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
        path_history  = [[] for _ in range(num_agents)]
        cbf_history   = [[] for _ in range(num_agents)]
    
        # --- Callback for rendering each physics step ---
        def render_callback(env_instance: NavEnv):
            # This function is only called if visualize is True.
            for j in range(num_agents):
                path_history[j].append((env_instance.robot_locations[j, 0], env_instance.robot_locations[j, 1]))
    
            # Create a list of agent pairs for connectivity visualization
            connectivity_pairs = []
            for i in range(env_instance.num_agent):
                pos1 = env_instance.robot_locations[i]
                if not env_instance.root_mask[i]:
                    parent_id = env_instance.connectivity_graph.get_parent(i)
                    pos2 = env_instance.robot_locations[parent_id]
                else:
                    pos2 = pos1
                connectivity_pairs.append((pos1, pos2))
    
    
            # Create a dictionary with visualization data
            viz_data = {
                "paths": path_history,
                "obs_local": env_instance.obstacle_states,
                "connectivity_pairs": connectivity_pairs,
                "target_local": env_instance.cbf_infos["nominal"]["p_targets"],
                "connectivity_trajs": env_instance.connectivity_traj,
            }
    
            # Call the draw_frame function
            draw_frame(ax1, ax2, env_instance, viz_data)
            
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf, dtype=np.uint8)[..., :3]
            frames.append(frame.copy())
    
        # Determine which callback to use
        callback_fn = render_callback
    
        # ================ Simulation Start ====================
        obs, _, info = env.reset(episode_index=eval_seed) # Use the fixed eval_seed
    
        # 초기 액션 세팅 + 스텝
        l = buffer.mini_step * buffer.mini_step_size
        h = (buffer.mini_step + 1) * buffer.mini_step_size
        buffer.obs[0][l:h].copy_(obs.view(1, *observation_space.shape))
        buffer.extras[0][l:h].copy_(info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate)
    
        ll, lh = l-buffer.mini_step_size, h-buffer.mini_step_size
        if lh == 0:
            lh = buffer.mini_step_size * buffer.num_rollout_blocks
        # Buffer의 Insert에서 이전 롤아웃의 마지막 상태를 현재 롤아웃의 첫 상태로 복사하는 로직에 맞춘 할당 (코드 처음에만 수행)
        buffer.obs[-1][ll:lh].copy_(buffer.obs[0][l:h])
        buffer.rec_states[-1][ll:lh].copy_(buffer.rec_states[0][l:h])
        buffer.extras[-1][ll:lh].copy_(buffer.extras[0][l:h])
    
        # act를 위한 변수 초기화
        rec_states = buffer.rec_states[0][l:h]
        mask = buffer.masks[0][l:h]
    
        total_reward = 0
        for step_num in range(steps):
    
            with torch.no_grad():
                values, actions, action_log_probs, \
                rec_states, action_maps             = agent.act(obs.view(1, *observation_space.shape),
                                                                rec_states,
                                                                mask,
                                                                info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate,
                                                                deterministic=True)
    
            # 시점 t+1에서의 observation, reward, done 추출
            next_obs, _, reward, terminated, truncated, next_info = env.step(actions, on_physics_step=callback_fn)
            done = torch.logical_or(terminated, truncated)
    
            for j in range(num_agents):
                # Obstacle distance for CBF plot
                obs_state = env.obstacle_states[j, :env.num_obstacles[j]]
                if env.num_obstacles[j] == 0:
                    min_dist = env.cfg.sensor_range**2
                else:
                    dist = np.linalg.norm(obs_state, axis=1)
                    min_ids = np.argmin(dist)
                    min_dist = obs_state[min_ids, 0]**2 + obs_state[min_ids, 1]**2
                
                # Connectivity distance for CBF plot
                p_c = env.cbf_infos["safety"]["p_c_agent"][j].reshape(-1)
                if len(p_c) > 0:
                    min_agent_dist = p_c[0]**2 + p_c[1]**2
                else:
                    min_agent_dist = 0
                
                agent_cbf_info = {"obs_avoid": min_dist - env.cfg.d_safe**2,
                                  "agent_conn": env.neighbor_radius**2 - min_agent_dist}
                cbf_history[j].append(agent_cbf_info)
    
            # 데이터 집계
            total_reward += reward.item()
    
            # 시점 transition
            obs = copy.deepcopy(next_obs)
            info = copy.deepcopy(next_info)
            mask = copy.deepcopy(~done)
            
            if done:
                break
    
        plt.close(fig)
        # --- Save GIF ---
        print(f"GIF saved at step {step_num+1}.")
        imageio.mimsave(gif_path, frames, fps=30)
    
        total_explorable_region = np.nonzero(env.map_info.gt == env.map_info.map_mask["free"])
        coverage_rate = 100 * max(1, env.prev_explored_region / (total_explorable_region.shape[0]))
    
        return total_reward, coverage_rate
    finally:
        # Restore original global RNG state
        np.random.set_state(rng_state_np)
        torch.set_rng_state(rng_state_torch)

if __name__ == '__main__':
    # Load config
    with open("config/nav_ppo_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # It's good practice to run tests with deterministic behavior
    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])

    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
    parser.add_argument("--map_type", type=str, default=None, choices=['corridor', 'maze', 'random', 'single_maze'], help="The type of the test map")
    parser.add_argument("--version", type=int, default=1, help="Verison of the model.")
    parser.add_argument("--seed", type=int, default=0, help="Seed number for randomization. Default is 0.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run for evaluation.")
    parser.add_argument("--visualize", action="store_true", help="Disable visualization to run faster.")


    args = parser.parse_args()
    
    # Run the test with evaluation
    run_simulation_test(args,
                        config,
                        steps=400, 
                        visualize=args.visualize,
                        num_episodes=args.episodes)
