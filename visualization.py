from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import List

# Map Legend Constants (from config)
FREE = 0
UNKNOWN = 1
OCCUPIED = 2
GOAL = 3
START = 4
FRONTIER = 5

# ======================================================================================
# Colormaps and Normalization (same as heuristic version)
# ======================================================================================
BELIEF_CMAP = colors.ListedColormap([
    '#FFFFFF',  # FREE
    '#BDBDBD',  # UNKNOWN
    '#000000',  # OCCUPIED
    "#DD8B86",  # GOAL
    "#97D8A8",  # START
    '#FF0000',  # FRONTIER
])
BELIEF_NORM = colors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5,4.5,5.5], BELIEF_CMAP.N)

GT_CMAP = colors.ListedColormap([
    '#FFFFFF',  # FREE
    '#FFFFFF',  # UNKNOWN (unused)
    '#000000',  # OCCUPIED
    "#DD8B86",  # GOAL
    "#97D8A8",  # START
    '#FFFFFF',  # FRONTIER (unused)
])
GT_NORM = BELIEF_NORM

AGENT_COLORS = ['#E6194B', '#4363D8', '#3CB44B', '#F58231', "#9D24C2", '#469990'] # Red, Blue, Green, Orange, Purple, Teal

def local_to_world(robot_pos: np.ndarray, local_points: np.ndarray, yaw: float) -> np.ndarray:
    """Transforms points from the robot's local frame to world coordinates."""
    if local_points.ndim == 1:
        local_points = local_points.reshape(1, -1)
        
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], 
                        [np.sin(yaw),  np.cos(yaw)]])
    
    world_points_relative = np.matmul(rot_mat, local_points.T).T
    world_points = world_points_relative + robot_pos
    return world_points


# ======================================================================================
# Drawing Functions (Adapted for RL Environment)
# ======================================================================================
def draw_frame(ax_gt, ax_belief, env, viz_data: dict):
    """
    Draws a single frame of the simulation for the RL environment.
    
    Args:
        ax_gt: Matplotlib axis for the ground truth map.
        ax_belief: Matplotlib axis for the belief map.
        env: The CBFEnv environment instance.
        viz_data: A dictionary containing visualization data like paths, commands, etc.
    """
    maps = env.map_info
    ax_gt.clear(); ax_belief.clear()
    ax_gt.imshow(maps.gt, cmap=GT_CMAP, norm=GT_NORM, origin='upper')
    ax_belief.imshow(maps.belief_frontier, cmap=BELIEF_CMAP, norm=BELIEF_NORM, origin='upper')

    def world_to_img(x, y):
        row, col = maps.world_to_grid(x, y)
        return col, row

    for i in range(env.num_agent):
        robot = env.robot_locations[i]
        robot_yaw = env.robot_angles[i]
        color = AGENT_COLORS[i % len(AGENT_COLORS)]

        # --- Safety and Connectivity Circles ---
        cx, cy = world_to_img(robot[0], robot[1])
        # Get safety/connectivity distances from the environment config
        radius_min_px = env.cfg.d_max * 0.5 / maps.res_m
        radius_max_px = env.neighbor_radius * 0.5 / maps.res_m

        for ax in (ax_gt, ax_belief):
            min_circle = plt.Circle((cx, cy), radius_min_px, color=color, fill=False, linestyle='--', linewidth=1, alpha=0.7)
            ax.add_patch(min_circle)
            max_circle = plt.Circle((cx, cy), radius_max_px, color=color, fill=False, linestyle='--', linewidth=1, alpha=0.7)
            ax.add_patch(max_circle)

        # --- Robot Center ---
        for ax in (ax_gt, ax_belief):
            center_dot = plt.Circle((cx, cy), 2, color=color, zorder=5)
            ax.add_patch(center_dot)

        # --- FOV sector (semi-transparent) ---
        half = math.radians(env.cfg.fov / 2.0)
        angles = np.linspace(-half, half, 20)
        poly_world = [(robot[0], robot[1])] + [
            (robot[0] + env.cfg.sensor_range * math.cos(robot_yaw + a),
             robot[1] + env.cfg.sensor_range * math.sin(robot_yaw + a)) for a in angles
        ]
        poly_img = [world_to_img(x, y) for (x, y) in poly_world]
        for ax in (ax_gt, ax_belief):
            ax.fill([p[0] for p in poly_img], [p[1] for p in poly_img],
                    alpha=0.3, color=color, zorder=2)

        # --- Target Points ---
        target_local = viz_data["target_local"][i]
        target_world = local_to_world(robot, target_local, robot_yaw)
        fx, fy = world_to_img(target_world[0, 0], target_world[0, 1])
        for ax in (ax_gt, ax_belief):
            ax.scatter(fx, fy, s=25, c=color, marker='.', zorder=4, alpha=0.4)

        # Obstacle
        obs_local_pos = viz_data["obs_local"][i, :env.num_obstacles[i]]
        if obs_local_pos.shape[0] > 0:
            obs_world_pos = local_to_world(robot, obs_local_pos, robot_yaw)
            fx, fy = zip(*[world_to_img(wx, wy) for wx, wy in obs_world_pos])
            for ax in (ax_gt, ax_belief):
                # Red
                ax.scatter(fx, fy, s=5, c=AGENT_COLORS[0], marker='.', zorder=4, alpha=0.4)
        
        # --- Path history ---
        path = viz_data["paths"][i]
        if len(path) > 1:
            xs, ys = zip(*[world_to_img(wx, wy) for wx, wy in path])
            for ax in (ax_gt, ax_belief):
                ax.plot(xs, ys, '-', linewidth=2, color=color, alpha=0.8, zorder=3)

        # # --- Robot heading/velocity arrow ---
        # v_cmd = np.sqrt(env.robot_velocities[i, 0]**2 + env.robot_velocities[i, 1]**2)
        # length = max(0.05, v_cmd * 0.5)
        # x2 = robot[0] + length * math.cos(robot_yaw)
        # y2 = robot[1] + length * math.sin(robot_yaw)
        # cx, cy = world_to_img(robot[0], robot[1])
        # cx2, cy2 = world_to_img(x2, y2)
        # for ax in (ax_gt, ax_belief):
        #     ax.arrow(cx, cy, cx2 - cx, cy2 - cy, head_width=5, head_length=8,
        #              fc=color, ec=color, length_includes_head=True, zorder=5)

        # --- Connectivity Lines ---
        connectivity_pairs = viz_data.get("connectivity_pairs", [])[i]
        pos1, pos2 = connectivity_pairs
        p1_img = world_to_img(pos1[0], pos1[1])
        p2_img = world_to_img(pos2[0], pos2[1])
        for ax in (ax_gt, ax_belief):
            ax.plot([p1_img[0], p2_img[0]], [p1_img[1], p2_img[1]], 
                    linestyle='--', color=color, linewidth=1.5, zorder=4)
        
        # --- Optimal Path ---
        connectivity_traj = viz_data.get("connectivity_trajs", [])[i]
        if len(connectivity_traj) > 0:
            xs, ys = zip(*[world_to_img(wx, wy) for wx, wy in connectivity_traj[0]])
            for ax in (ax_gt, ax_belief):
                ax.plot(xs, ys, '-', linewidth=2, color=color, alpha=0.8, zorder=3)
        
    # --- Global Region ---
    regions = env.regions
    valid_regions = env.valid_regions
    for region in regions:
        r0, r1, c0, c1 = map(int, region.bounds)
        x0, y0 = maps.grid_to_world(r0, c0)  # (row_min, col_min)
        x1, y1 = maps.grid_to_world(r1, c1)  # (row_max, col_max)
        # 월드 -> 이미지 좌표
        p0x, p0y = world_to_img(x0, y0)
        p1x, p1y = world_to_img(x1, y1)
        rx, ry = min(p0x, p1x), min(p0y, p1y)
        rw, rh = abs(p1x - p0x), abs(p1y - p0y)
        if not region in valid_regions:
            edgecolor = 'k'
        else:
            edgecolor = 'y'
            for ax in (ax_gt, ax_belief):
                # 레이블 표시
                ax.text(rx, ry - 6, "Valid Region", color=edgecolor, fontsize=6, weight='bold', zorder=6)

        for ax in (ax_gt, ax_belief):
            rect = plt.Rectangle(
                (rx, ry), rw, rh,
                fill=False,
                edgecolor=edgecolor, 
                linestyle='--',
                linewidth=1.0,
                zorder=7,
                alpha=0.9
            )
            ax.add_patch(rect)
            ax.add_patch(plt.Rectangle(
                (rx, ry), rw, rh,
                fill=True,
                facecolor=edgecolor,
                alpha=0.1,
                linewidth=0,
                zorder=6
            ))
    
    # --- Local Region ---
    # cluster_cmap = plt.get_cmap('tab20')
    # num_colors = len(cluster_cmap.colors)
    cluster_infos = env.cluster_infos
    for label, value in cluster_infos.items():
        regions = value["regions"]

        for region in regions:
            r0, r1, c0, c1 = map(int, region)
            x0, y0 = maps.grid_to_world(r0, c0) 
            x1, y1 = maps.grid_to_world(r1, c1)
            p0x, p0y = world_to_img(x0, y0)
            p1x, p1y = world_to_img(x1, y1)
            rx, ry = min(p0x, p1x), min(p0y, p1y)
            rw, rh = abs(p1x - p0x), abs(p1y - p0y)

            # color_index = label % num_colors
            edgecolor = 'r'
            for ax in (ax_gt, ax_belief):
                rect = plt.Rectangle(
                    (rx, ry), rw, rh,
                    fill=False,
                    edgecolor=edgecolor,    
                    linestyle='-',
                    linewidth=1.0,
                    zorder=5,
                    alpha=0.9
                )
                ax.add_patch(rect)

    # --- Final Touches ---
    for ax in (ax_gt, ax_belief):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Ground Truth' if ax is ax_gt else 'Belief')

# ======================================================================================
# Plotting Functions (Copied from heuristic, should be compatible)
# ======================================================================================

def plot_agent_distances(paths: List[List[tuple[float, float]]], d_safe: float, d_max: float, dt: float, save_path: str = None):
    num_agents = len(paths)
    if num_agents < 2:
        return
    min_len = min(len(p) for p in paths)
    timesteps = np.arange(min_len) * dt
    fig, axes = plt.subplots(num_agents, 1, figsize=(10, 2 * num_agents), sharex=True)
    if num_agents == 1: axes = [axes]
    fig.suptitle('Inter-Agent Distances Over Time', fontsize=16)
    for i in range(num_agents):
        ax = axes[i]
        path_i = np.array(paths[i][:min_len])
        for j in range(num_agents):
            if i == j: continue
            path_j = np.array(paths[j][:min_len])
            distances = np.linalg.norm(path_i - path_j, axis=1)
            ax.plot(timesteps, distances, label=f'Distance to Agent {j}', color=AGENT_COLORS[j % len(AGENT_COLORS)])
        ax.axhline(y=d_safe, color='r', linestyle='--', label=f'd_safe ({d_safe}m)')
        ax.axhline(y=d_max, color='b', linestyle=':', label=f'd_max ({d_max}m)')
        ax.set_title(f'Agent {i}'); ax.set_ylabel('Distance (m)'); ax.legend(loc='upper right'); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path); plt.close(fig)
    else: plt.show()

def plot_control_inputs(nominal_history, safe_history, dt, num_agents, save_path: str = None):
    if not nominal_history or not safe_history: return
    timesteps = np.arange(len(nominal_history[0])) * dt
    fig, axes = plt.subplots(num_agents, 2, figsize=(12, 3 * num_agents), sharex=True)
    if num_agents == 1: axes = np.array([axes])
    fig.suptitle('Nominal vs. Safe Control Inputs', fontsize=16)
    for i in range(num_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        nom_a, nom_w = zip(*nominal_history[i])
        safe_a, safe_w = zip(*safe_history[i])
        ax_v = axes[i, 0]
        ax_v.plot(timesteps, nom_a, '--', color=color, label='v_nominal'); ax_v.plot(timesteps, safe_a, '-', color=color, label='v_safe')
        ax_v.set_title(f'Agent {i} - Linear Acceleration'); ax_v.set_ylabel('v (m/s)'); ax_v.legend(); ax_v.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_w = axes[i, 1]
        ax_w.plot(timesteps, nom_w, '--', color=color, label='w_nominal'); ax_w.plot(timesteps, safe_w, '-', color=color, label='w_safe')
        ax_w.set_title(f'Agent {i} - Angular Velocity'); ax_w.set_ylabel('w (rad/s)'); ax_w.legend(); ax_w.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path); plt.close(fig)
    else: plt.show()

def plot_obs_dists(obs_state, dt, num_agents, save_path: str = None):
    if not obs_state or not obs_state: return
    timesteps = np.arange(len(obs_state[0])) * dt
    fig, axes = plt.subplots(num_agents, 2, figsize=(12, 3 * num_agents), sharex=True)
    if num_agents == 1: axes = np.array([axes])
    fig.suptitle('Nominal vs. Safe Control Inputs', fontsize=16)
    for i in range(num_agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        obs_x, obs_y = zip(*obs_state[i])
        ax_x = axes[i, 0]
        ax_x.plot(timesteps, obs_x, '-', color=color, label='x_local');
        ax_x.set_title(f'Agent {i} - Obstacle X Distance'); ax_x.set_ylabel('x (m)'); ax_x.legend(); ax_x.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_y = axes[i, 1]
        ax_y.plot(timesteps, obs_y, '-', color=color, label='y_local');
        ax_y.set_title(f'Agent {i} - Obstacle Y Distance'); ax_y.set_ylabel('y (m)'); ax_y.legend(); ax_y.grid(True, which='both', linestyle='--', linewidth=0.5)
    axes[-1, 0].set_xlabel('Time (s)'); axes[-1, 1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path); plt.close(fig)
    else: plt.show()

def plot_cbf_values(cbf_history: List[List[dict]], dt: float, num_agents: int, save_path: str = None):
    if not cbf_history or not cbf_history[0]: return
    min_len = min(len(h) for h in cbf_history)
    timesteps = np.arange(min_len) * dt
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3 * num_agents), sharex=True)
    if num_agents == 1: axes = [axes]
    fig.suptitle('Control Barrier Function (h) Values Over Time', fontsize=16)
    for i in range(num_agents):
        ax = axes[i]
        history_i = cbf_history[i][:min_len]
        h_obs_avoid = [h.get('obs_avoid', [0]) if h.get('obs_avoid') is not None else 0 for h in history_i]
        h_agent_avoid = [min(h.get('agent_avoid', [0])) if h.get('agent_avoid') is not None and len(h.get('agent_avoid')) > 0 else 0 for h in history_i]
        h_agent_conn = [h.get('agent_conn', [0]) if h.get('agent_conn') is not None else 0 for h in history_i]
        ax.plot(timesteps, h_obs_avoid, label='h_obs_avoid (min)', linestyle='--')
        ax.plot(timesteps, h_agent_avoid, label='h_agent_avoid (min)', linestyle='--')
        ax.plot(timesteps, h_agent_conn, label='h_agent_conn (min)', linestyle=':')
        ax.axhline(y=0, color='r', linestyle='-', linewidth=1.5, label='h=0 (Safety Boundary)')
        ax.set_title(f'Agent {i}'); ax.set_ylabel('h value'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.set_ylim(bottom=-0.1)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path); plt.close(fig)
    else: plt.show()

def plot_psi_values(cbf_history: List[List[dict]], dt: float, num_agents: int, save_path: str = None):
    if not cbf_history or not cbf_history[0]: return
    min_len = min(len(h) for h in cbf_history)
    timesteps = np.arange(min_len) * dt
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3 * num_agents), sharex=True)
    if num_agents == 1: axes = [axes]
    fig.suptitle('HOCBF Constraint (psi) Values Over Time', fontsize=16)
    for i in range(num_agents):
        ax = axes[i]
        history_i = cbf_history[i][:min_len]
        psi_agent_avoid = [min(h.get('psi_agent_avoid', [0])) if h.get('psi_agent_avoid') is not None and len(h.get('psi_agent_avoid')) > 0 else 0 for h in history_i]
        psi_agent_conn = [min(h.get('psi_agent_conn', [0])) if h.get('psi_agent_conn') is not None and len(h.get('psi_agent_conn')) > 0 else 0 for h in history_i]
        ax.plot(timesteps, psi_agent_avoid, label='psi_agent_avoid (min)', linestyle='--')
        ax.plot(timesteps, psi_agent_conn, label='psi_agent_conn (min)', linestyle=':')
        ax.axhline(y=0, color='r', linestyle='-', linewidth=1.5, label='psi=0 (Constraint Boundary)')
        ax.set_title(f'Agent {i}'); ax.set_ylabel('psi value'); ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5); ax.set_ylim(bottom=-0.5)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path: plt.savefig(save_path); plt.close(fig)
    else: plt.show()