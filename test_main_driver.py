import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import traceback
from typing import List

# Assuming the necessary classes are importable from your project structure
from task.env.nav_env import NavEnv
from task.agent.sac import SACAgent
from task.utils import get_nominal_control, world_to_local
from visualization import draw_frame, plot_cbf_values

PATH = os.path.join(os.getcwd())
SPECIFIC_PATH = ""


def create_models(cfg: dict, obs_dim: int, state_dim: int, action_dim: int, device: torch.device) -> dict:
    """
    Helper function to create models based on the config.
    """
    model_cfg = cfg['model']

    return

def run_simulation_test(cfg: dict, steps: int, out_dir: str = 'test_results'):
    """
    Runs a simulation test for the CBFEnv and SACAgent, generating a GIF and plots.
    """
    print("=== Starting Simulation Test ===")
    # --- Output Directory ---
    os.makedirs(out_dir, exist_ok=True)
    print(f"Results will be saved to: {out_dir}")
    # --- Device ---
    device = torch.device(cfg['env']['device'])
    # --- Environment ---
    env = NavEnv(episode_index=0, device=device, cfg=cfg['env'])
    print("Environment created and reset.")
    # --- Agent & Models ---
    obs_dim = env.cfg.num_obs
    state_dim = env.cfg.num_state
    action_dim = env.cfg.num_act
    num_agents = cfg['env']['num_agent']

    # models = create_models(cfg, obs_dim, state_dim, action_dim, device)
    # agent = SACAgent(num_agents=num_agents, models=models, device=device, cfg=cfg['agent'])
    # print("Agent with models created.")

    # checkpoint_path = os.path.join(PATH, SPECIFIC_PATH)
    # agent.load(checkpoint_path)
    # print(f"Load Checkpoint at {checkpoint_path}")

    # --- Simulation Loop ---
    frames: List[np.ndarray] = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6))

    # Data tracking for plots
    nominal_inputs_history = [[] for _ in range(num_agents)]
    safe_inputs_history    = [[] for _ in range(num_agents)]
    min_obs_state = [[] for _ in range(num_agents)]
    path_history  = [[] for _ in range(num_agents)]
    cbf_history   = [[] for _ in range(num_agents)]
    connectivity_pairs = []
    demo = False
    # try:
    #     obs, state, info = env.reset(episode_index=10)
    #     for step_num in range(steps):
    #         if demo:
    #             demo_nominal = get_nominal_control(p_target=info["nominal"]["p_targets"],
    #                                                on_search=info["nominal"]["on_search"],
    #                                                v_current=info["safety"]["v_current"],
    #                                                a_max=env.max_lin_acc,
    #                                                w_max=env.max_ang_vel,
    #                                                v_max=env.max_lin_vel)
    #             raw_actions = torch.tensor(demo_nominal / 
    #                                         np.array([env.max_lin_acc, 
    #                                                   env.max_ang_vel])).to(device=device)
                
    #             actions, feasible = agent.safety(raw_actions, info["safety"])

    #         else:
    #             raw_actions, actions, _, feasible = agent.act(obs, safety_info=info["safety"], deterministic=True)

    #         # --- Environment steps ---
    #         next_obs, next_state, reward, terminated, truncated, next_info = env.step(actions)
    #         done = np.any(terminated) or np.any(truncated)
            
    #         # # --- Record data ---
    #         for j in range(num_agents):
    #             # Obstacle
    #             obs_state = env.obstacle_states[j, :env.num_obstacles[j]]
    #             if env.num_obstacles[j] == 0:
    #                 min_obs_state[j].append(np.array([0.3, 0.3]))

    #                 min_dist = 0.3**2
    #             else:
    #                 dist = np.linalg.norm(obs_state, axis=1)
    #                 min_ids = np.argmin(dist)
    #                 min_dist = obs_state[min_ids, 0]**2 + obs_state[min_ids, 1]**2
    #                 min_obs_state[j].append(obs_state[min_ids].copy())
                
    #             # Connetctivity
    #             p_c = next_info["safety"]["p_c_agent"][j].reshape(-1)
    #             if len(p_c) > 0:
    #                 min_agent_dist = p_c[0]**2 + p_c[1]**2
    #             else:
    #                 min_agent_dist = 0

    #             agent_cbf_info = {"obs_avoid": min_dist-env.cfg.d_obs**2,
    #                               "agent_conn": env.neighbor_radius**2 - min_agent_dist}

    #             path_history[j].append((env.robot_locations[j, 0], env.robot_locations[j, 1]))
    #             nominal_inputs_history[j].append(raw_actions[j].cpu().numpy())
    #             safe_inputs_history[j].append(actions[j].cpu().numpy())
    #             cbf_history[j].append(agent_cbf_info)

    #         print(f"Step: {step_num + 1}/{steps} | Done: {done} | # of Clusters : {len(env.cluster_infos.keys())}")

    #         # # --- Visualization ---
    #         # Create a list of agent pairs for connectivity visualization
    #         connectivity_pairs = []
    #         for i in range(env.num_agent):
    #             pos1 = env.robot_locations[i]
    #             if not env.root_mask[i]:
    #                 parent_id = env.connectivity_graph.get_parent(i)
    #                 pos2 = env.robot_locations[parent_id]
    #             else:
    #                 pos2 = pos1
    #             connectivity_pairs.append((pos1, pos2))
                

    #         # Create a dictionary with visualization data
    #         viz_data = {
    #             "paths": path_history,
    #             "obs_local": env.obstacle_states,
    #             "last_cmds": [(actions[i][0], actions[i][1]) for i in range(num_agents)],
    #             "connectivity_pairs": connectivity_pairs, # Add pairs to viz_data
    #             "target_local": info["nominal"]["p_targets"],
    #             "connectivity_trajs": env.connectivity_traj,
    #         }

    #         # Call the new draw_frame function
    #         draw_frame(ax1, ax2, env, viz_data)
            
    #         fig.canvas.draw()
    #         buf = fig.canvas.buffer_rgba()
    #         frame = np.asarray(buf, dtype=np.uint8)[..., :3]
    #         frames.append(frame.copy())

    #         obs = next_obs
    #         state = next_state
    #         info = next_info

    #         if done:
    #             print("Simulation ended because 'done' is True.")
    #             break

    # except Exception as e:
    #     print(f"{e}")
    #     traceback.print_exc()

    # finally:
    #     plt.close(fig)
    #     # --- Save GIF ---
    #     gif_path = os.path.join(out_dir, 'simulation_test.gif')
    #     print(f"Saving GIF to {gif_path}...")
    #     imageio.mimsave(gif_path, frames, fps=30)
    #     print("GIF saved.")
    #     # # --- Generate and Save Plots ---
    #     plot_cbf_values(cbf_history, env.dt, num_agents, save_path=os.path.join(out_dir, 'cbf_values.png'))

    print("=== Simulation Test Finished ===")


if __name__ == '__main__':
    # Load config
    with open("config/nav_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # It's good practice to run tests with deterministic behavior
    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])
    
    # Run the test
    run_simulation_test(config, steps=1000)