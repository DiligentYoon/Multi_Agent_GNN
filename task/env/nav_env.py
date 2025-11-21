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
        self.cbf_infos = {}
        self.cbf_infos["safety"] = {}
        self.cbf_infos["nominal"] = {}
    

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
        """
        제어 입력을 계산하기 위한 전처리 작업 수행
        1. Target Point action을 받아서 각 에이전트에게 할당
        2. Leader Agent 지정
            Inputs:
                actions : 에이전트 별 Target Point
        """
        return super()._pre_apply_action(actions)


    def _apply_actions(self):
        """
        모든 Agent의 제어 입력 (선가속도, 각속도)을 계산하고, 제어 입력에 따른 State Update
        """
        active_mask = ~self.reached_goal.squeeze()

        # 1. 속도 업데이트 (선속도)
        self.robot_speeds[active_mask] += self.preprocessed_actions[active_mask, 0] * self.dt
        self.robot_speeds[active_mask] = np.clip(self.robot_speeds[active_mask], 0.0, self.max_lin_vel)
        
        # 2. 위치 업데이트
        current_angles = self.robot_angles[active_mask]
        speeds = self.robot_speeds[active_mask]
        self.robot_locations[active_mask, 0] += speeds * np.cos(current_angles).flatten() * self.dt
        self.robot_locations[active_mask, 1] += speeds * np.sin(current_angles).flatten() * self.dt
        
        # 3. 각도 업데이트
        yaw_rates = np.clip(self.preprocessed_actions[active_mask, 1], -self.max_ang_vel, self.max_ang_vel)
        self.robot_yaw_rate[active_mask] = yaw_rates.reshape(-1, 1)
        new_angles = ((current_angles + self.robot_yaw_rate[active_mask] * self.dt + np.pi) % (2 * np.pi)) - np.pi
        self.robot_angles[active_mask] = new_angles
        
        # 4. World Frame 속도 벡터 업데이트
        self.robot_velocities[active_mask, 0] = self.robot_speeds[active_mask] * np.cos(new_angles).flatten()
        self.robot_velocities[active_mask, 1] = self.robot_speeds[active_mask] * np.sin(new_angles).flatten()


    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        특정 종료조건 및 타임아웃 계산

            Return :
                1. terminated : 
                    1-1. 벽에 충돌
                    1-2. 드론끼리 충돌
                    1-3. 골 지점 도달
                2. truncated :
                    2-1. 타임아웃
        """
        # Planning State 업데이트
        self._compute_intermediate_values()

        # ============== Done 계산 로직 ===================

        # ---- Truncated 계산 -----
        timeout = self.num_step >= self.max_episode_steps - 1
        truncated = np.full((self.num_agent, 1), timeout, dtype=np.bool_)

        # ---- Terminated 계산 ----
        cells = self.map_info.world_to_grid_np(self.robot_locations)
        rows, cols = cells[:, 1], cells[:, 0]

        # 목표 도달 유무 체크
        self.is_reached_goal = (self.map_info.gt[rows, cols] == self.map_info.map_mask["goal"]).reshape(-1, 1)

        # 맵 경계 체크
        H, W = self.map_info.H, self.map_info.W
        out_of_bounds = (rows < 0) | (rows >= H) | (cols < 0) | (cols >= W)

        # 유효한 셀에 대해서만 값 확인
        valid_indices = ~out_of_bounds
        valid_rows, valid_cols = rows[valid_indices], cols[valid_indices]

        # 장애물 충돌 (맵 밖 포함)
        hit_obstacle = np.zeros_like(out_of_bounds, dtype=np.bool_)
        hit_obstacle[valid_indices] = self.map_info.gt[valid_rows, valid_cols] == self.map_info.map_mask["occupied"]
        self.is_collided_obstacle = (hit_obstacle | out_of_bounds)[:, np.newaxis]

        # 드론 간 충돌 (점유 셀이 겹치면 충돌 판단)
        flat_indices = rows * W + cols
        unique_indices, counts = np.unique(flat_indices, return_counts=True)
        collided_indices = unique_indices[counts > 1]
        
        self.is_collided_drone.fill(False)
        for idx in collided_indices:
            colliding_agents = np.where(flat_indices == idx)[0]
            for agent_idx in colliding_agents:
                self.is_collided_drone[agent_idx] = True

        # 개별 로봇이 충돌하거나 목표에 도달하면 종료
        terminated = self.is_collided_obstacle | self.is_collided_drone | self.is_reached_goal

        return terminated, truncated, self.is_reached_goal


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
        Initial State를 스폰 기준에 맞게 세팅

            Inputs:
                max_attempts (int): 최대 시도 횟수.

            Returns:
                Tuple[np.ndarray, np.ndarray]: (world_x, world_y).
        """
        H, W = self.map_info.H, self.map_info.W
        d_max, d_min = self.cfg.d_conn, self.cfg.d_safe 

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
                continue

            # d_max 제약조건 검사 (개별)
            min_distances_to_neighbors = np.min(dist_matrix, axis=1)
            is_connected = np.all(min_distances_to_neighbors <= d_max)
            if not is_connected:
                continue

            print(f"Valid start positions found after {attempt + 1} attempts.")
            return selected_world_coords[:, 0], selected_world_coords[:, 1]

        raise RuntimeError(f"No valid starting positions found within {max_attempts}")
    
    def update_cbf_infos(self, agent_id: int):
        """
        HOCBF Safety Filter에 필요한 정보를 각 Agent에 대해서 업데이트
        : Safety -> [장애물 정보, 이웃 에이전트 정보, 자식 에이전트 정보]
        : Nominal -> [타겟 위치]
            
            Inputs:
                agent_id: 에이전트 고유 ID
            
        """
        i = agent_id
        pos_i = self.robot_locations[i]
        yaw_i = self.robot_angles[i]
        pass