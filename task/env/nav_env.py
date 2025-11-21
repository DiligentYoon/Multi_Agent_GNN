import numpy as np
import torch

from typing import Tuple, List
from task.base.env.env import Env
from task.controller.hocbf import DifferentiableHOCBFLayer
from task.graph.graph import ConnectivityGraph
from task.planner.astar import *
from task.utils import *

from task.graph.kdtree import RegionKDTree
from task.graph.tree_utils import *


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
        self.global_kd_tree = RegionKDTree((0, self.map_info.H, 0, self.map_info.W), valid_threshold=0.05)
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        self.local_frontiers = np.zeros((self.num_agent, self.cfg.num_rays, 2), dtype=np.float32)
        self.root_mask = np.zeros(self.num_agent, dtype=np.int_)
        self.connectivity_graph = ConnectivityGraph(self.num_agent)
        self.connectivity_traj = [[] for _ in range(self.num_agent)]
        self.num_obstacles = np.zeros(self.num_agent, dtype=np.int_)
        self.num_neighbors = (self.num_agent-1) * np.ones(self.num_agent, dtype=np.int_)

        self.obstacle_states = np.zeros((self.num_agent, self.cfg.max_obs, 2), dtype=np.float32)
        self.neighbor_states = np.zeros((self.num_agent, self.cfg.max_agents-1, 4), dtype=np.float32)
        self.neighbor_ids = np.zeros((self.num_agent, self.cfg.max_agents-1), dtype=np.int_)

        # Replanning State
        self.agent_paths = [None] * self.num_agent
        self.agent_path_targets = [None] * self.num_agent

        # Done flags
        self.is_collided_obstacle = np.zeros((self.num_agent, 1), dtype=np.bool_)
        self.is_collided_drone = np.zeros((self.num_agent, 1), dtype=np.bool_)

        # Additional Info
        self.cbf_infos = {}
        self.cbf_infos["safety"] = {}
        self.cbf_infos["nominal"] = {}
    
        # TODO: deleted later
        self.num_frontiers = np.zeros(self.num_agent, dtype=np.int_)
        self.total_dt = 0


    def reset(self, episode_index: int = None):
        """
        Reset Episode with Map Change

            Inputs:
                episode_index : seed value for map randomization

            Returns:
                obs : observation vector
                state : state vector
                info : additional information
        """
        self.connectivity_traj = [[] for _ in range(self.num_agent)]
        self.robot_speeds = np.zeros(self.num_agent, dtype=np.float32)
        
        # Reset replanning states
        self.agent_paths = [None] * self.num_agent
        self.agent_path_targets = [None] * self.num_agent

        obs, state, info = super().reset(episode_index)

        return obs, state, info
    
    
    def _pre_apply_action(self, actions):
        """
        [Centralized] 제어 입력을 계산하기 위한 전처리 작업 수행
        1. Target Point action을 받아서 각 에이전트에게 할당
        2. MST 업데이트 with Leader Agent 지정

            Inputs:
                actions : 에이전트 별 Target Point
        """
        pass


    def _apply_actions(self):
        """
        [Decentralized] 모든 Agent의 제어 입력 (선가속도, 각속도)을 계산하고, 제어 입력에 따른 State Update
        """
        # Per-step CBF Info 업데이트
        self.update_cbf_infos()
        # 제어 입력 계산
        control_inputs, feasible = self.controller(self.cbf_infos["nominal"], self.cbf_infos["safety"])
        active_mask = ~self.reached_goal.squeeze()

        # 속도 업데이트 (선속도)
        speeds = self.robot_speeds[active_mask] + control_inputs[active_mask, 0] * self.dt
        self.robot_speeds[active_mask] = np.clip(speeds, 0.0, self.max_lin_vel)
        
        # 위치 업데이트
        current_angles = self.robot_angles[active_mask]
        self.robot_locations[active_mask, 0] += self.robot_speeds[active_mask] * np.cos(current_angles) * self.dt
        self.robot_locations[active_mask, 1] += self.robot_speeds[active_mask] * np.sin(current_angles) * self.dt
        
        # 각도 업데이트
        yaw_rates = np.clip(control_inputs[active_mask, 1], -self.max_ang_vel, self.max_ang_vel)
        new_angles = ((current_angles + yaw_rates * self.dt + np.pi) % (2 * np.pi)) - np.pi
        self.robot_yaw_rate[active_mask] = yaw_rates
        self.robot_angles[active_mask] = new_angles
        
        # World Frame 속도 벡터 업데이트
        self.robot_velocities[active_mask, 0] = self.robot_speeds[active_mask] * np.cos(new_angles)
        self.robot_velocities[active_mask, 1] = self.robot_speeds[active_mask] * np.sin(new_angles)
    

    def _get_observations(self) -> np.ndarray | list[dict]:
        """
        """
        return None
    

    def _get_states(self) -> np.ndarray | list[dict]:
        """
        """
        return copy.deepcopy(self.obs_buf)
    

    def _get_rewards(self):
        """
        """
        pass


    def _get_dones(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        [Centralized] 특정 종료조건 및 타임아웃 계산

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
        reached_goal = (self.map_info.gt[rows, cols] == self.map_info.map_mask["goal"]).reshape(-1, 1)

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
        terminated = self.is_collided_obstacle | self.is_collided_drone | reached_goal

        return terminated, truncated, reached_goal


    def _compute_intermediate_values(self):
        """
        [Centralized] 업데이트된 state값들을 바탕으로, Planning state 계산
        : Frontier Graph
        : Robot Graph
        """
        # Global Frontier Marking
        self.map_info.belief_frontier, dt = global_frontier_marking(self.map_info)
        self.total_dt += dt
        
        # Frontier Graph Construction

        # Robot Graph Construction

        # ====================== TODO: This part is deleted later. Only for test =======================
        total_local_regions = []
        total_local_scores = []
        frontier_cells = [[] for _ in range(self.num_agent)]
        for i in range(self.num_agent):
            local_frontiers, frontiers_cell = self.detect_frontier(agent_id=i)

            bel = self.map_info.belief
            bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]
            for r, c in frontiers_cell:
                if bel[r, c] == self.map_info.map_mask["free"]: bel[r, c] = self.map_info.map_mask["frontier"]
            
            # Store frontier information
            num_frontiers = min(len(local_frontiers), self.cfg.num_rays)
            if num_frontiers > 0:
                self.local_frontiers[i, :num_frontiers] = local_frontiers
            else:
                self.local_frontiers[i, :] = 0
            self.num_frontiers[i] = num_frontiers
            frontier_cells[i].append(frontiers_cell)
        bel[bel == self.map_info.map_mask["frontier"]] = self.map_info.map_mask["free"]

        root_id = np.argmax(self.num_frontiers)
        self.root_mask.fill(0)
        self.root_mask[root_id] = 1
        self.connectivity_graph.update_and_compute_mst(self.robot_locations, root_id)


        # 1) Global KD-Tree Valid Node 선택 & 점수 할당
        regions = self.global_kd_tree.leaves
        valid_regions = self.global_kd_tree.update_node_states(self.map_info, self.robot_locations)
        # 2) Local KD-Tree 빌드 & 점수 할당
        for region in valid_regions:
            local_regions, local_scores = split_and_score_local_region(region.bounds, self.robot_locations, self.map_info)
            total_local_regions.extend(local_regions)
            total_local_scores.extend(local_scores)
        # 3) Local Region Clustering w.r.t each score
        if total_local_regions:
            cluster_infos = local_region_clustering(total_local_regions, total_local_scores)
        else:
            # 예외 처리 필요
            raise ValueError("No valid local regions found.")
        # 4) Target Point Generation
        targets_rc, targets_prob, valid_cluster_ids = sample_k_targets_in_multi_regions_value(self.map_info, 
                                                                                              cluster_infos,
                                                                                              k=self.num_agent*3, 
                                                                                              rng=self.seed)
        # 5) Point Allocation by Hungarian algorithms
        assigned_rc = assign_targets_hungarian(self.map_info, self.robot_locations, targets_rc, self.num_agent)
        # 6) Memory for cascade system
        self.regions = regions
        self.valid_regions = valid_regions
        self.assigned_rc = np.array(assigned_rc)

        # =================================================================================================


    def _update_infos(self):
        pass



    # =========== Auxilary Functions =============

    def _set_init_state(self,
                        max_attempts: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        [Centralized] Initial State를 스폰 기준에 맞게 세팅

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
    


    def detect_frontier(self, 
                        agent_id: int) -> tuple[np.ndarray, np.ndarray]:
        """
            Raycast in FOV, update belief FREE/OCCUPIED 
                Inputs:
                    - agent_id: Agent Numbering
                Return:
                    - frontier_local:  [(lx, ly), ...]  
                    - frontier_rc:     [(row, col), ...] 
                    - obs_local:       [(lx, ly), ...] 
        """
        drone_pose = np.hstack((self.robot_locations[agent_id], self.robot_angles[agent_id]))
        maps = self.map_info
        H, W = maps.H, maps.W
        half = math.radians(self.fov / 2.0)
        angles = np.linspace(-half, half, self.cfg.num_rays)
    
        frontier_local: List[Tuple[float, float]] = []
        frontier_rc: List[Tuple[int, int]] = []
    
        for a in angles:
            ang = drone_pose[2] + a
            step = maps.res_m
            L = int(self.sensor_range / step)
    
            last_rc = None
            hit_recorded = False          # per-ray: obs 최대 1개
            frontier_candidate_rc = None  # per-ray: frontier 후보(마지막 FREE∧UNKNOWN-인접)
    
            for i in range(1, L + 1):
                x = drone_pose[0] + i * step * math.cos(ang)
                y = drone_pose[1] + i * step * math.sin(ang)
                if x < 0 or y < 0 or x > maps.meters_w or y > maps.meters_h:
                    break
    
                r, c = maps.world_to_grid(x, y)
                if last_rc == (r, c):
                    continue
                last_rc = (r, c)
    
                if maps.gt[r, c] == maps.map_mask["occupied"]:
                    # 첫 OCC 히트만 기록
                    break  # 이 ray 종료 (더 이상 진행 X)
    
                else:
                    # 관측된 FREE 갱신
                    if maps.belief[r, c] != maps.map_mask["start"]:
                        maps.belief[r, c] = maps.map_mask["free"]
    
                    # 이 셀의 8-이웃 중 UNKNOWN이 있으면 'frontier 후보'
                    found_unknown = False
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr = r + dr; cc = c + dc
                            if 0 <= rr < H and 0 <= cc < W and maps.belief[rr, cc] == maps.map_mask["unknown"]:
                                found_unknown = True
                                break
                        if found_unknown:
                            break
    
                    # frontier 후보는 ray를 따라 '마지막으로' 갱신하여, 경계에 가장 가까운 FREE를 선택
                    if found_unknown:
                        frontier_candidate_rc = (r, c)
    
            # ray가 끝난 뒤, 후보가 있으면 frontier를 1개만 최종 채택
            if frontier_candidate_rc is not None:
                r, c = frontier_candidate_rc
                wx, wy = maps.grid_to_world(r, c)
                dx = wx - drone_pose[0]; dy = wy - drone_pose[1]
                cth = math.cos(-drone_pose[2]); sth = math.sin(-drone_pose[2])
                lx = cth*dx - sth*dy; ly = sth*dx + cth*dy
                frontier_local.append((lx, ly))
                frontier_rc.append((r, c))
    
        return np.array(frontier_local), np.array(frontier_rc)




    def update_neighbor_info(self, agent_id: int, robot_pos: np.ndarray):
        """
        [Decentralized] Neighbor 기준을 바탕으로 주변 이웃 에이전트 정보 업데이트

            Inputs:
                agent_id: 에이전트 고유 ID
                robot_pos: 특정 에이전트 Pose [x, y, yaw]
        """
        robot_pos_i = robot_pos[agent_id]
        other_pos = np.delete(robot_pos, agent_id, axis=0)
        other_vel = np.delete(self.robot_velocities, agent_id, axis=0)
        # lx, ly
        rel_pos = world_to_local(w1=robot_pos_i[:2], w2=other_pos[:, :2], yaw=robot_pos_i[2])
        # v_jx, v_jy
        rel_vel = world_to_local(w1=None, w2=other_vel, yaw=robot_pos_i[2])
        # Neighbor Ids
        distance = np.linalg.norm(rel_pos, axis=1)
        neighbor_ids = np.where(distance <= self.neighbor_radius)[0]
        # Neighbor State for Agent Collision Avoidance
        self.num_neighbors[agent_id] = len(neighbor_ids)
        if self.num_neighbors[agent_id] > 0:
            self.neighbor_states[agent_id, :self.num_neighbors[agent_id], :2] = rel_pos[neighbor_ids]
            self.neighbor_states[agent_id, :self.num_neighbors[agent_id], 2:] = rel_vel[neighbor_ids]
        else:
            # 0개인 경우 (Connectivity Slack으로 인한 예외상황) : 가장 가까운 Agent가 neighbors
            closest_neighbor_id = np.argmin(distance)
            self.num_neighbors[agent_id] = 1
            self.neighbor_states[agent_id, 0, :2] = rel_pos[closest_neighbor_id]
            self.neighbor_states[agent_id, 0, 2:] = rel_vel[closest_neighbor_id]


    def update_obstacle_info(self, agent_id: int, robot_pos:np.ndarray):
        """
        [Decentralized] Ray-casting 방식을 통해 Obstacle 정보 업데이트

            Inputs:
                agent_id: 에이전트 고유 ID
                robot_pos: 특정 에이전트 Pose [x, y, yaw]
        """
        local_obstacles = sense(robot_pos, 
                                self.map_info,
                                self.cfg.sensor_range,
                                self.cfg.num_rays,
                                self.cfg.fov)
        
        # Store obstacle information
        num_obs = min(len(local_obstacles), self.cfg.max_obs)
        if num_obs > 0:
            self.obstacle_states[agent_id, :num_obs] = local_obstacles[:num_obs]
        else:
            self.obstacle_states[agent_id, :] = 0
        self.num_obstacles[agent_id] = num_obs


    def update_cbf_infos(self):
        """
        [Decentralized] HOCBF Safety Filter에 필요한 정보를 각 Agent에 대해서 업데이트
        : Safety -> [장애물 정보, 이웃 에이전트 정보, 자식 에이전트 정보]
        : Nominal -> [타겟 위치]
            
            Inputs:
                agent_id: 에이전트 고유 ID
        """
        self.neighbor_states.fill(0)
        self.obstacle_states.fill(0)  
        on_conn_list = []
        target_pos_list =[]
        p_obs_list = []
        p_agents_list = []
        p_c_agent_list = []
        v_c_agent_list = []
        v_agents_list = []
        connectivity_traj = [[] for _ in range(self.num_agent)]
        
        # Aegnt-wise CBF Info Update
        robot_pos = np.hstack((self.robot_locations, self.robot_angles.reshape(-1, 1)))
        for i in range(self.num_agent):
            pos_i = self.robot_locations[i]
            yaw_i = self.robot_angles[i]
            # Obstacle Info
            self.update_obstacle_info(agent_id=i, robot_pos=robot_pos[i])
            num_obs = self.num_obstacles[i]
            p_obs_list.append(self.obstacle_states[i, :num_obs])
            # Neighbor Agent Info
            self.update_neighbor_info(agent_id=i, robot_pos=robot_pos)
            num_neighbors = self.num_neighbors[i]
            p_agents_list.append(self.neighbor_states[i, :num_neighbors, :2])
            v_agents_list.append(self.neighbor_states[i, :num_neighbors, 2:])
            # ==== Target Agent Info ====
            # Root Node : Parent Node 존재 X
            # Leaf Node : CHild Node 존재 X
            # Reciprocal Connectivity Relationship:
            #   1. Parent Node : Child Node와 HOCBF 제약
            #   2. Child Node : Parent Node까지 A* Optimal Path
            #   3. Root Node : Child Node에 대해서 Only HOCBF제약
            #   4. Leaf Node : Parents Node에 대해서 Only A* Optimal Path
            parent_id = self.connectivity_graph.get_parent(i)
            child_id = self.connectivity_graph.get_child(i)
            if parent_id == -1:
                # Root Node : Only HOCBF because No Parent
                pos_i_op = self.robot_locations[i]
                pos_i_c = self.robot_locations[child_id]
                vel_i_c = self.robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i-pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)
            elif child_id is None:
                # Leaf Node : Only A* because No child
                pos_i_op = self.robot_locations[parent_id]
                pos_i_c = np.array([])
                vel_i_c = np.array([])
            else:
                # Other Node : Both A* and HOCBF
                pos_i_op = self.robot_locations[parent_id]
                pos_i_c = self.robot_locations[child_id]
                vel_i_c = self.robot_velocities[child_id]
                if pos_i_c.shape[0] > 1:
                    # MST는 여러개의 자식노드가 있을 수 있음 -> Closest로 지정
                    min_ids = np.argmin(np.linalg.norm(pos_i-pos_i_c, axis=1))
                    pos_i_cbf = pos_i_c[min_ids]
                    vel_i_cbf = vel_i_c[min_ids]
                else:
                    pos_i_cbf = pos_i_c.reshape(-1)
                    vel_i_cbf = vel_i_c.reshape(-1)
            
            # Control Barrier Function Info for Backward Connectivity
            p_p = world_to_local(w1=pos_i, w2=pos_i_op, yaw=yaw_i)
            p_c = world_to_local(w1=pos_i, w2=pos_i_cbf, yaw=yaw_i) if child_id is not None else np.array([])
            v_c = world_to_local(w1=None, w2=vel_i_cbf, yaw=yaw_i) if child_id is not None else np.array([]) # [NOTE] HOCBF Formulation 상 좌표계 정렬된 절대속도 필요

            p_c_agent_list.append(p_c)
            v_c_agent_list.append(v_c)

            # Forward Connectivity Target Info
            min_dist = np.linalg.norm(p_p)
            if (min_dist < self.cfg.d_conn) or (parent_id == -1):
                # Connectivity 유지 중인 Parent Node & Parent가 없는 Root Node
                on_conn = False
                start_cell = self.map_info.world_to_grid_np(pos_i) # (col, row)
                end_cell = np.flip(self.assigned_rc[i]) # (col, row)
            else:
                # Connectivity가 끊길 위험이 있는 Child Node & Child가 없는 Leaf Node 
                on_conn = True
                start_cell = self.map_info.world_to_grid_np(pos_i) # (col, row)
                end_cell = self.map_info.world_to_grid_np(pos_i_op) # (col, row)
            

            # ==== Path Planning ====
            replan = False
            current_path_cells = self.agent_paths[i]
            current_target_cell = self.agent_path_targets[i]

            # 1. 맨 처음이거나 경로가 없는 경우
            if current_path_cells is None:
                replan = True
            # 2. 타겟 포인트가 바뀐 경우
            elif not np.array_equal(end_cell, current_target_cell):
                replan = True
            # 3. 기존 경로가 더 이상 유효하지 않은 경우 (장애물 충돌)
            elif not is_path_valid(self.map_info, np.array(current_path_cells)):
                replan = True

            if replan:
                path_cells = astar_search(self.map_info,
                                          start_pos=np.flip(start_cell),
                                          end_pos=np.flip(end_cell),
                                          agent_id=i)               # (row, col)
                if path_cells is not None and len(path_cells) > 0:
                    self.agent_paths[i] = path_cells                # (row, col)
                    self.agent_path_targets[i] = end_cell           # (row, col)
                else:
                    # 경로 생성 실패 시, 현재 위치 고정
                    self.agent_paths[i] = np.flip(start_cell)
                    self.agent_path_targets[i] = start_cell
                
                path_cells = self.agent_paths[i]

            else:
                # 기존 경로 유지
                path_cells = self.agent_paths[i]
                if len(path_cells) > 1:
                    path_world_coords = self.map_info.grid_to_world_np(np.flip(path_cells, axis=1)) # (row, col) -> (col, row) -> (x, y)
                    distances = np.linalg.norm(path_world_coords - pos_i, axis=1)
                    closest_idx = np.argmin(distances)
                    # 가장 가까운 지점부터 끝까지의 경로를 새로운 경로로 사용
                    path_cells = path_cells[closest_idx:]
                    self.agent_paths[i] = path_cells

            if path_cells is not None and len(path_cells) > 0:
                # Path 생성된 경우
                optimal_traj = self.map_info.grid_to_world_np(np.flip(path_cells, axis=1)) # (row, col) -> (col, row) -> (x, y)
            else:
                # Path 없는 경우
                optimal_traj = np.array([self.robot_locations[i]])
            optimal_traj_local = world_to_local(w1=pos_i, w2=optimal_traj, yaw=yaw_i)
            distances = np.linalg.norm(optimal_traj_local, axis=1)
            ids = np.where(distances >= self.cfg.d_safe*2)[0]
            if len(ids) > 0:
                target_pos = optimal_traj_local[ids[0]]
            else:
                target_pos = optimal_traj_local[-1]

            connectivity_traj[i].append(optimal_traj)
            target_pos_list.append(target_pos)
            on_conn_list.append(on_conn)
                
        self.connectivity_traj = connectivity_traj

        self.cbf_infos["safety"] = {
            "v_current": self.robot_speeds.tolist(),
            "p_obs": p_obs_list,
            "p_agents": p_agents_list,
            "v_agents": v_agents_list,
            "p_c_agent": p_c_agent_list,
            "v_c_agent": v_c_agent_list
        }

        self.cbf_infos["nominal"] = {
            "p_targets" : target_pos_list,
            "on_conn" : on_conn_list
        }

            
