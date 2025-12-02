import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import gymnasium as gym
import datetime

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from typing import List
from task.env.nav_env import NavEnv
from task.agent.sac import SACAgent
from task.agent.ppo import Agent, PPOAgent
from task.model.models import RL_ActorCritic
from task.buffer.rolloutbuffer import RolloutBuffer, CoMappingRolloutBuffer

from visualization import draw_frame, plot_cbf_values

PATH = os.path.join(os.getcwd())
SPECIFIC_PATH = ""


def create_model(cfg: dict, observation_space: gym.Space, action_space: gym.Space, device: torch.device) -> RL_ActorCritic:
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

    actor_critic = RL_ActorCritic(observation_space.shape, action_space,
                                  model_type=model_cfg['model_type'],
                                  base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                               'use_history': model_cfg['use_history'],
                                               'ablation': model_cfg['ablation']},
                                  lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                  eps=model_cfg['eps']).to(device)
    return actor_critic


def create_agent(cfg: dict, model: RL_ActorCritic, num_agents: int, eval_freq: int, 
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
                                        rec_state_size=model.rec_state_size,
                                        extras_size=num_agents * 6
                                        ).to(device)
        agent = PPOAgent(model, device, agent_cfg)
    elif algorithm == "SAC":
        raise ValueError("Not Buffer Implementation yet.")
        agent = SACAgent()
    else:
        raise ValueError("Unvalid Algorithm Types.")

    return agent, buffer


def run_simulation_test(cfg: dict, steps: int, out_dir: str = 'test_results', visualize: bool = True, load_file_path: str = None):
    """
    Runs a simulation test, generating a GIF and plots.
        Inputs:
            cfg: Total Configuration
            stpes: maximum simulation steps
            out_dir: visualization save directory
            visualize : If True, generates and saves a GIF and plots. 
                        If False, runs the simulation without visualization.
    """
    print("=== Starting Simulation Test ===")
    if not visualize:
        print("Visualization is OFF.")
        
    # --- Output Directory & Writer & Checkpoint ---
    # os.makedirs(out_dir, exist_ok=True)
    print(f"Evaluation Results will be saved to: {out_dir}")
    start_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")

    experiment_dir = os.path.join("results", f"{start_time}_{cfg['agent']['experiment']['directory']}")
    # os.makedirs(experiment_dir, exist_ok=True)
    print(f"Training Results will be saved to {experiment_dir}")

    timesteps = cfg["train"]["timesteps"]
    write_interval = cfg['agent']['experiment']['write_interval']
    if write_interval == 'auto':
        write_interval = int(timesteps / 10)
    writer = SummaryWriter(log_dir=experiment_dir)

    checkpoint_interval = cfg['agent']['experiment']['checkpoint_interval']
    if checkpoint_interval == 'auto':
        checkpoint_interval = int(timesteps / 10)
    # --- Device ---
    device = torch.device(cfg['env']['device'])
    # --- Environment ---
    env = NavEnv(episode_index=0, device=device, cfg=cfg['env'])
    print("Environment created and reset.")
    # --- Agent & Models ---
    pr = env.cfg.pooling_downsampling_rate
    num_agents = cfg['env']['num_agent']
    observation_space = gym.spaces.Box(0, 1, (8 + num_agents, env.map_info.H // pr, env.map_info.W // pr), dtype='uint8')
    action_space = gym.spaces.Box(0, (env.map_info.H // pr) * (env.map_info.W // pr) - 1, (num_agents,), dtype='int32')
    actor_critic_model = create_model(cfg['model'], observation_space, action_space, device)
    agent, buffer      = create_agent(cfg['agent'], actor_critic_model, num_agents,
                                      cfg['train']['eval_freq'], observation_space, action_space, device)
    if load_file_path is not None:
        agent.model.load(os.path.join(os.getcwd(), load_file_path), device=device)

    # --- Visualization and Data Tracking Setup ---
    frames: List[np.ndarray] = []
    fig, ax1, ax2 = None, None, None
    if visualize:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))

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
    callback_fn = render_callback if visualize else None

    # ================ Simulation Start ====================
    obs, _, info = env.reset(episode_index=25)

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

        # 시점 t+1에서의 observation, rec_states, addtional_info 
        # 시점 t에서의 action 저장
        # 시점 t+1에서의 reward, done 저장
        buffer.insert(
            # obs_t+1, rec_state_t+1
            next_obs.view(1, *observation_space.shape), rec_states,
            # action_t, action_log_t, value_t
            actions, action_log_probs, values,
            # reward_t+1, done_t+1
            reward, ~done,
            # info_t+1
            next_info["additional_obs"].view(1, -1) // env.cfg.pooling_downsampling_rate
        )

        print(f"Frontier Index: {actions.cpu().numpy()}")
        print(f"Reward : {reward.item():.2f}")
        print(f'# of frontier (obs) : { torch.nonzero(next_obs[1, :, :]).shape[0] }')

        # --- Record data for final plots (once per RL step) ---
        if visualize:
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

        print(f"RL Step: {step_num+1}/{steps}")

        # if (step_num % buffer.rollouts == buffer.rollouts - 1) and buffer.mini_step == 0:
        #     # 롤아웃이 끝났으면 파라미터 업데이트 수행. 먼저 마지막 스텝에 대한 +1 step value값 예측
        #     print("Updating agent...")
        #     next_value = agent.model.get_value(
        #         buffer.obs[-1],
        #         buffer.rec_states[-1],
        #         buffer.masks[-1],
        #         extras=buffer.extras[-1]
        #     )[0].detach()

        #     buffer.compute_returns(next_value, True, agent.cfg['discount_factor'], agent.cfg['gae_lambda'])

        #     value_loss, action_loss, dist_entropy = agent.update(buffer)
        #     if value_loss > 0:
        #             value_losses.append(value_loss)
        #             action_losses.append(action_loss)
        #             dist_entropies.append(dist_entropy)
        #     buffer.after_update()
        #     print("Update completed.")

        # TODO: 추후, Ray 전용 모델 저장 로직으로 사용
        # if step_num > 0 and (step_num - episode_length) // write_interval < step_num // write_interval:
        #     write_tracking_data(step_num)
        # # Checkpoint
        # if step_num > 0 and (step_num - episode_length) // checkpoint_interval < step_num // checkpoint_interval:
        #     save_checkpoint(step_num)


        # if step_num > 0 and (step_num % write_interval):
        #     agent.model.save(step_num)
        # Checkpoint
        # if step_num > 0 and (step_num & checkpoint_interval):
        #     filepath = os.path.join(os.getcwd(), os.path.join(experiment_dir, "checkpoints", f"agent_{step_num}.pt"))
        #     agent.model.save(filepath)


        # 시점 transition
        obs = next_obs
        info = next_info
        mask = ~done
        
        if done:
            print("Episode finished.")
            break

    if visualize:
        plt.close(fig)
        # --- Save GIF ---
        gif_path = os.path.join(out_dir, 'simulation_test.gif')
        print(f"Saving GIF to {gif_path}...")
        imageio.mimsave(gif_path, frames, fps=20)
        print("GIF saved.")
        
        # --- Generate and Save Plots ---
        plot_cbf_values(cbf_history, env.dt * env.decimation, num_agents, save_path=os.path.join(out_dir, 'cbf_values.png'))

    print("=== Simulation Test Finished ===")




if __name__ == '__main__':
    # Load config
    with open("config/nav_ppo_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # It's good practice to run tests with deterministic behavior
    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])
    
    # Run the test with visualization enabled
    run_simulation_test(config, 
                        steps=30, 
                        visualize=True,
                        load_file_path='results/25-12-01_14-57-47_MARL/checkpoints/agent_188160.pt')