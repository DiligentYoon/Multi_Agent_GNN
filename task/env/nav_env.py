import numpy as np
import torch

from typing import Tuple, List
from task.base.env.env import Env
from task.controller.hocbf import DifferentiableHOCBFLayer
from task.graph.graph import ConnectivityGraph
from task.graph.graph_utils import *
from task.utils import *


from .nav_env_cfg import NavEnvCfg


class NavEnv(Env):
    def __init__(self, episode_index: int | np.ndarray, device: torch.device, cfg: dict):
        self.cfg = NavEnvCfg(cfg)
        super().__init__(self.cfg)

        # Simulation Parameters
        self.device = device
        self.seed = episode_index
        self.dt = self.cfg.physics_dt
        self.decimation = self.cfg.decimation
        self.neighbor_radius = self.cfg.d_conn * 1.5
        self.max_episode_steps = self.cfg.max_episode_steps

        # Controller
        self.cfg.controller['a_max'] = self.max_lin_acc
        self.cfg.controller['w_max'] = self.max_ang_vel
        self.cfg.controller['v_max'] = self.max_lin_vel
        self.cfg.controller['d_max'] = self.neighbor_radius
        self.cfg.controller['d_safe'] = self.cfg.d_safe
        self.cfg.controller['max_agents'] = self.cfg.max_agents
        self.cfg.controller['max_obs'] = self.cfg.max_obs
        self.controller = DifferentiableHOCBFLayer(cfg=self.cfg.controller, device=self.device)

        # Planning State
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        self.local_frontiers = np.zeros((self.num_agent, self.cfg.num_rays, 2), dtype=np.float32)
        self.connectivity_graph = ConnectivityGraph(self.num_agent)
        self.connectivity_traj = [[] for _ in range(self.num_agent)]
        self.num_obstacles = np.zeros(self.num_agent, dtype=np.int_)
        self.num_neighbors = (self.num_agent-1) * np.ones(self.num_agent, dtype=np.int_)

        self.obstacle_states = np.zeros((self.num_agent, self.cfg.max_obs, 2), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.cfg.max_agents-1, 4), dtype=np.float32)
        self.neighbor_ids = np.zeros((self.num_agent, self.cfg.max_agents-1), dtype=np.int_)

        # Done flags
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_reached_goal = np.zeros((self.num_agent, 1), dtype=np.bool_)

        # Additional Info
        self.infos["safety"] = {}
        self.infos["nominal"] = {}
    

    def reset(self, episode_index: int = None):
        """
        Reset Episode with Map Change

            Inputs:
                episode_index : seed value for map randomization

            Returns:
                obs : observation vector
                state : state vector
                info : additional information [safety info, nominal control info]
        """
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        obs, state, info = super().reset(episode_index)

        return obs, state, info
    
    
    def _pre_apply_action(self, actions):
        return super()._pre_apply_action(actions)


    def _apply_action(self, agent_id):
        """
        각 Agent의 제어 입력 (선가속도, 각속도)에 따라, 속도 및 위치 업데이트

            Inputs:
                agent_id : 에이전트 고유 ID
        """
        i = agent_id
        self.robot_speeds[i] += self.preprocessed_actions[i, 0] * self.dt
        self.robot_speeds[i] = np.clip(self.robot_speeds[i], 0.0, self.max_lin_vel)
        # Non-Holodemic Model 특성에 의해 Position 먼저 업데이트
        self.robot_locations[i, 0] += self.robot_speeds[i] * np.cos(self.robot_angles[i]) * self.dt
        self.robot_locations[i, 1] += self.robot_speeds[i] * np.sin(self.robot_angles[i]) * self.dt
        # Yaw rate를 바탕으로 각도 업데이트
        self.robot_yaw_rate[i] = np.clip(self.preprocessed_actions[i, 1], -self.max_ang_vel, self.max_ang_vel)
        self.robot_angles[i] = ((self.robot_angles[i] + self.robot_yaw_rate[i] * self.dt + np.pi) % (2 * np.pi)) - np.pi
        # World Frame 속도 세팅
        self.robot_velocities[i, 0] = self.robot_speeds[i] * np.cos(self.robot_angles[i])
        self.robot_velocities[i, 1] = self.robot_speeds[i] * np.sin(self.robot_angles[i])


    def _compute_intermediate_values(self):
        """
        업데이트된 state값들을 바탕으로, Planning state 계산
        """
        drone_pos = np.hstack((self.robot_locations, self.robot_angles.reshape(-1, 1)))

        # --- Zero-Padding Initialization ---
        self.neighbor_states.fill(0)
        self.obstacle_states.fill(0)  
        self.local_frontiers.fill(0)
        self.neighbor_ids.fill(0)
        frontier_cells = [[] for _ in range(self.num_agent)]
        # -----------------------------------

        for i in range(self.num_agent):
            # Pos
            drone_pos_i = drone_pos[i]
            # lx, ly
            rel_pos = world_to_local(w1=drone_pos_i[:2], w2=drone_pos[:, :2], yaw=drone_pos_i[2])
            # v_jx, v_jy
            rel_vel = world_to_local(w1=None, w2=self.robot_velocities, yaw=drone_pos_i[2])
            # sqrt(lx^2 + lx^2)
            distance = np.linalg.norm(rel_pos, axis=1)
            # [Decentralized] 자기자신 제외한 모든 Agent Global Ids
            other_agent_ids = np.where(distance > 1e-5)[0]
            # [Decentralized] Local Graph 반경에 포함된 이웃 Agent Global Ids
            neighbor_agent_ids = np.where(np.logical_and(distance <= self.neighbor_radius, distance > 1e-5))[0]

            # Neighbor State for Agent Collision Avoidance
            self.num_neighbors[i] = len(neighbor_agent_ids)
            if self.num_neighbors[i] > 0:
                self.neighbor_states[i, :self.num_neighbors[i], :2] = rel_pos[neighbor_agent_ids]
                self.neighbor_states[i, :self.num_neighbors[i], 2:] = rel_vel[neighbor_agent_ids]
                self.neighbor_ids[i, :self.num_neighbors[i]] = neighbor_agent_ids # Global Ids
            else:
                # 0개인 경우 (Connectivity Slack으로 인한 예외상황) : 가장 가까운 Agent가 neighbors
                closest_neighbor_id_local = np.argmin(distance[other_agent_ids])
                closest_neighbor_id_global = other_agent_ids[closest_neighbor_id_local]
                self.num_neighbors[i] = 1
                self.neighbor_states[i, 0, :2] = rel_pos[closest_neighbor_id_global]
                self.neighbor_states[i, 0, 2:] = rel_vel[closest_neighbor_id_global]
                self.neighbor_ids[i, 0] = closest_neighbor_id_global # Global Ids

            # Obstacles & Frontiers Sensing
            local_frontiers, frontiers_cell, local_obstacles = self.sense_and_update(agent_id=i)

            bel = self.map_info.belief
            bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]
            for r, c in frontiers_cell:
                if bel[r, c] == self.map_info.map_mask["free"]: bel[r, c] = self.map_info.map_mask["frontier"]

            # Store obstacle information
            num_obs = min(len(local_obstacles), self.cfg.max_obs)
            if num_obs > 0:
                self.obstacle_states[i, :num_obs] = local_obstacles[:num_obs]
            else:
                self.obstacle_states[i, :] = 0
            self.num_obstacles[i] = num_obs
            # Store frontier information
            num_frontiers = min(len(local_frontiers), self.cfg.num_rays)
            if num_frontiers > 0:
                self.local_frontiers[i, :num_frontiers] = local_frontiers
            else:
                self.local_frontiers[i, :] = 0
            self.num_frontiers[i] = num_frontiers
            frontier_cells[i].append(frontiers_cell)
        
        # 마지막 루프에 대한 Frontier Marking 정리
        bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]

        # Update된 Belief Map에 대한 Frontier 마킹
        reset_flag = np.all(self.map_info.belief_frontier == self.map_info.map_mask["unknown"])
        self.map_info.belief_frontier = global_frontier_marking(self.map_info, reset_flag, frontier_cells)




    # =========== Auxilary Functions =============

    def _set_init_state(self,
                        max_attempts: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initial State를 기준에 맞게 세팅

            Inputs:
                max_attempts (int): 최대 시도 횟수.

            Returns:
                Tuple[np.ndarray, np.ndarray]: (world_x, world_y).
        """
        H, W = self.map_info.H, self.map_info.W
        d_max, d_min = self.cfg.d_max, self.cfg.d_safe 

        start_rows, start_cols = np.where(np.logical_and(self.map_info.gt == self.map_info.map_mask["start"],
                                                            self.map_info.gt != self.map_info.map_mask["occupied"]))
        # 변환 함수의 일관성을 위해, [col, row] 순서로 일치
        start_cell_candidates = np.stack([start_cols, start_rows], axis=1)

        if len(start_cell_candidates) < self.num_agent:
            raise ValueError(f"Number of start cells ({len(start_cell_candidates)}) is less than "
                            f"the number of agents ({self.num_agent}).")

        for attempt in range(max_attempts):
            # 랜덤 샘플링
            indices = np.random.choice(len(start_cell_candidates), self.num_agent, replace=False)
            selected_cells = start_cell_candidates[indices] # shape: (num_agent, 2)

            # 월드 좌표로 변환
            selected_world_coords = self.map_info.grid_to_world_np(selected_cells)

            diff = selected_world_coords[:, np.newaxis, :] - selected_world_coords[np.newaxis, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)

            np.fill_diagonal(dist_matrix, np.inf)

            # d_safe 제약조건 검사 (전체)
            is_safe = np.min(dist_matrix) > d_min
            if not is_safe:
                continue # 조건을 만족하지 않으면 다시 샘플링

            # d_max 제약조건 검사 (개별)
            min_distances_to_neighbors = np.min(dist_matrix, axis=1)
            is_connected = np.all(min_distances_to_neighbors <= d_max)
            if not is_connected:
                continue

            print(f"Valid start positions found after {attempt + 1} attempts.")
            return selected_world_coords[:, 0], selected_world_coords[:, 1]

        raise RuntimeError(f"No valid starting positions found within {max_attempts}")