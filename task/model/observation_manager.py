import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn

from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from task.utils import distance_field


def get_local_map_boundaries(agent_location, local_map_size, global_map_size):
    agent_location_r, agent_location_c = agent_location
    local_map_w, local_map_h = local_map_size
    global_map_w, global_map_h = global_map_size

    if local_map_size != global_map_size:
        gc1, gr1 = agent_location_c - local_map_w // 2, agent_location_r - local_map_h // 2
        gc2, gr2 = gc1 + local_map_w, gr1 + local_map_h
        if gc1 < 0:
            gc1, gc2 = 0, local_map_w
        if gc2 > global_map_w:
            gc1, gc2 = global_map_w - local_map_w, global_map_w

        if gr1 < 0:
            gr1, gr2 = 0, local_map_h
        if gr2 > global_map_h:
            gr1, gr2 = global_map_h - local_map_h, global_map_h
    else:
        gc1, gc2, gr1, gr2 = 0, global_map_w, 0, global_map_h

    return [gc1, gc2, gr1, gr2]


class ObservationManager:
    def __init__(self, cfg, device):
        self.num_robots = cfg["num_robots"]
        self.unit_size_m = cfg["unit_size_m"]
        self.real_map_w = cfg['real_map_w']
        self.real_map_h = cfg['real_map_h']
        self.global_map_w = cfg["global_map_w"]
        self.global_map_h = cfg["global_map_h"]
        self.local_map_w = cfg["local_map_w"]
        self.local_map_h = cfg["local_map_h"]
        self.pooling_downsampling = cfg["pooling_downsampling"]
        self.device = device

        # Initializing full map, down-scaled map, info
        # obs/frontier/all pos/all trajectory/explored/explorable/history pos/history goal
        self.global_map_size = max(self.global_map_h, self.global_map_w)
        self.global_map = torch.zeros(8, self.global_map_size, self.global_map_size).float().to(device)
        self.row_offset = self.global_map_size - self.real_map_h
        # 1-2 cartesian global agent location, 3-6 local map boundary
        self.global_info = torch.zeros(self.num_robots, 6).long().to(device)
        

        # Initial full and local pose
        self.global_pose = np.zeros((self.num_robots, 2)) # [x ,y]
        self.local_pose = np.zeros((self.num_robots, 2))  # [x, y]

        # Origin of local map
        self.local_map_origins = np.zeros((self.num_robots, 2)) # [x, y]

        # Local Map Boundaries (min x & y, max x & y in the global map)
        self.local_map_boundary = np.zeros((self.num_robots, 4)).astype(np.int32) # [cmin, cmax, rmin, rmax]


    def world_to_grid_np(self, world: np.ndarray) -> np.ndarray:
        H, W = self.global_map_size, self.global_map_size
        if world.ndim == 1:
            world = world.reshape(1, -1)
        x = world[:, 0]
        y = world[:, 1]
        col = np.clip(x / self.unit_size_m, 0, W - 1).astype(np.long)
        row = (H - 1) - (np.clip(y / self.unit_size_m, 0, H - 1)).astype(np.long)

        grid_position = np.stack((col, row), axis=-1)

        return grid_position


    def grid_to_world_np(self, grid: np.ndarray) -> np.ndarray:
        H, W = self.global_map_size, self.global_map_size
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        col = grid[:, 0]
        row = grid[:, 1]
        x = col * self.unit_size_m
        y = (H - 1 - row) * self.unit_size_m    

        world = np.stack((x, y), axis=-1)

        return world


    def init_map_and_pose(self, cell_pos: np.ndarray, cartesian_pos: np.ndarray):
        """
        Initialization Observation Manager Each Reset Phase.

            Inputs:
                cell_pos: [col, row]
                cartesian_pos: [x, y]
        """
        self.global_map.fill_(0.)
        self.global_pose[:] = cartesian_pos
        agent_location = cell_pos # [col, row]
        for e in range(self.num_robots):
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            self.global_info[e, :2] = torch.tensor((agent_location_r, agent_location_c))
            self.global_map[3, agent_location_r, agent_location_c] = 1.

            self.local_map_boundary[e] = get_local_map_boundaries((agent_location_r, agent_location_c), 
                                                                  (self.local_map_w, self.local_map_h), 
                                                                  (self.global_map_size, self.global_map_size)) # [col, row]
            self.local_map_origins[e] = self.grid_to_world_np(np.array([self.local_map_boundary[e, 0], 
                                                                        self.local_map_boundary[e, 2]])) # [col, row] -> [x, y]
        self.local_pose = self.global_pose - self.local_map_origins # [x, y]

    
    def update_global(self,
                      robot_pos: np.ndarray,
                      obstacle: np.ndarray, 
                      frontier: np.ndarray, 
                      explored: np.ndarray, 
                      explorable: np.ndarray) -> torch.Tensor:
        """
        Global Map & Global Pose 업데이트 수행 (Global Map에서 누적의 의미를 갖는 채널들도 업데이트)

            Inputs:
                robot_pos: [x, y]
                obstacle: Binary Obstacle Map
                frontier: Binary Frontier Map
                explored: Binary Explored Map
                explorable: Binary Explorable Map
            Returns:
                Global_Input : Global Obs [8 + Num_Agent, H, W]
        """
        if not np.all(obstacle.shape == self.global_map[0].shape):
            current_h, current_w = obstacle.shape # (100, 500)
            target_h, target_w = self.global_map.shape[1], self.global_map.shape[2] # (500, 500)

            valid_obstacle = np.zeros((target_h, target_w))
            valid_frontier = np.zeros((target_h, target_w))
            valid_explored = np.zeros((target_h, target_w))
            valid_explorable = np.zeros((target_h, target_w))

            # 맨 왼쪽아래에 일관적으로 배치
            start_row = self.row_offset
            end_col = current_w
            
            valid_obstacle[start_row:, :end_col] = obstacle
            valid_frontier[start_row:, :end_col] = frontier
            valid_explored[start_row:, :end_col] = explored
            valid_explorable[start_row:, :end_col] = explorable
        else:
            valid_obstacle = obstacle
            valid_frontier = frontier
            valid_explored = explored
            valid_explorable = explorable

        
        self.global_map[0, :, :] = torch.from_numpy(valid_obstacle).float()
        self.global_map[1, :, :] = torch.from_numpy(valid_frontier).float()
        self.global_map[4, :, :] = torch.from_numpy(valid_explored).float()
        self.global_map[5, :, :] = torch.from_numpy(valid_explorable).float()
        self.global_map[2, :, :].fill_(0.)

        lmb = self.local_map_boundary # [col, row]
        self.global_pose = robot_pos # [x, y]
        agent_location = self.world_to_grid_np(self.global_pose) # [col, row]
        for e in range(self.num_robots):
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            lmb[e] = get_local_map_boundaries((agent_location_r, agent_location_c), 
                                              (self.local_map_w, self.local_map_h), 
                                              (self.global_map_size, self.global_map_size)) # [col, row]
            agent_location_r = max(0, min(self.global_map_size, agent_location_r))
            agent_location_c = max(0, min(self.global_map_size, agent_location_c))
            self.global_info[e, :2] = torch.tensor((agent_location_r, agent_location_c))
            self.global_map[[2, 6], agent_location_r, agent_location_c] = 1
            self.global_map[3, agent_location_r, agent_location_c] = 1
            self.local_map_origins[e] = self.grid_to_world_np(np.array([lmb[e, 0],
                                                                       lmb[e, 2]])) # [col, row] -> [x, y]
            
        self.global_info[:, 2:] = torch.from_numpy(self.local_map_boundary)
        self.local_pose = self.global_pose - self.local_map_origins


    def get_global_input(self, g_history: torch.Tensor) -> torch.Tensor:
        """
        
        """
        global_input = nn.MaxPool2d(self.pooling_downsampling)(self.global_map)
        # frontier 중, obstacle과 겹치는 부분은 삭제
        global_input[1, :, :][global_input[0, :, :].bool()] = 0
        # frontier selection의 난이도를 낮추기 위한 clustering 수행
        global_input[1, :, :] = self.clustering_frontier_map(global_input[1, :, :].cpu().numpy()).to(self.device)
        global_input[6, :, :] -= global_input[2, :, :]
        global_input[7, :, :] = g_history
        dist_input = torch.zeros((self.num_robots, self.global_map_size, self.global_map_size))
        obstacle = self.global_map[0, :, :].bool()

        rows = obstacle.any(1).cpu().numpy()
        cols = obstacle.any(0).cpu().numpy()
        obstacle = obstacle.cpu()

        for i in range(self.num_robots):
            agent_cell_pos = self.world_to_grid_np(self.global_pose[i, :2]).reshape(-1) # [col, row]
            dist_input[i, agent_cell_pos[1], agent_cell_pos[0]] = 1
            row = np.copy(rows)
            col = np.copy(cols)
            row[agent_cell_pos[1]] = True
            col[agent_cell_pos[0]] = True
            distance_field(dist_input[i, :, :], obstacle, optimized=(row, col))

        dist_input = dist_input.to(self.device)
        dist_input[self.global_map[1:2, :, :].repeat(self.num_robots, 1, 1) == 0] = 4
        for i in range(self.num_robots):
            agent_cell_pos = self.world_to_grid_np(self.global_pose[i, :2]).reshape(-1)
            dist_input[i, agent_cell_pos[1], agent_cell_pos[0]] = 4
        dist_input = -nn.MaxPool2d(self.pooling_downsampling)(-dist_input)
        dist_input[dist_input > 4] = 4
        global_input = torch.cat((global_input, dist_input), dim=0) # dim: [8 + Num_Agent, H, W]

        return global_input
    

    def clustering_frontier_map(self, 
                                frontier_map: np.ndarray, 
                                eps: float = 3, 
                                min_samples: int = 4) -> torch.Tensor:
        """
        Frontier Map에서 마킹된 Frontier들을 대상으로 클러스터링 수행

            Inputs:
                frontier_map: Binary Frontier Map
                eps: DBSCAN eps 파라미터이며, 픽셀 단위로 설정
                min_samples: DBSCAN min_samples 파라미터
            Returns:
                frontier_clusters: Pytorch Tensor, (N, 2) shape의 유효한 Frontier Cluster 좌표 (row, col)
        """
        refined_map = torch.zeros_like(torch.from_numpy(frontier_map)).long()
        frontiers = np.argwhere(frontier_map == 1)
        if len(frontiers) == 0:
            return refined_map
        
        # DBSCAN 수행
        try:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(frontiers)
        except:
            db = DBSCAN(eps=eps, min_samples=1).fit(frontiers)
        unique_labels = set(db.labels_)

        # 노이즈만 존재하는 최악의 경우, 노이즈라도 할당하기
        if len(unique_labels) == 1 and -1 in unique_labels:
            refined_map[frontiers[:, 0], frontiers[:, 1]] = 1
            return refined_map
        
        # DBSCAN으로 생성된 각 클러스터에 대해 재귀적 기하 분할 수행
        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = frontiers[db.labels_ == label]
            refined_targets = np.array(self._recursive_split(cluster_points))
            refined_map[refined_targets[:, 0], refined_targets[:, 1]] = 1
        
        return refined_map

    def _get_medoid(self, points: np.ndarray) -> np.ndarray:
            """
            [Helper] 점들의 평균(Mean)을 구하고, 그 평균과 가장 가까운 '실제 점(Medoid)'을 반환
            """
            # 1. 무게 중심(Mean) 계산
            centroid = points.mean(axis=0)
            
            # 2. 모든 점들과 무게 중심 사이의 유클리드 거리 계산
            # axis=1을 따라 norm을 구함 -> (N,)
            distances = np.linalg.norm(points - centroid, axis=1)
            
            # 3. 거리가 가장 짧은 점의 인덱스 찾기
            nearest_idx = np.argmin(distances)
            
            # 4. 해당 점 반환
            return points[nearest_idx]


    def _recursive_split(self, points: np.ndarray, max_extent: float = 4, min_split_points: int = 4) -> list:
        """
        재귀적으로 클러스터를 검사하고 PCA 기반 K-Means로 분할
        """
        # 한 클러스터 내에 점이 너무 적으면 더 이상 쪼개지 않음 
        if len(points) < min_split_points:
            return [self._get_medoid(points)]

        # 클러스터 내 Point들로 기하적 크기 검사
        r_len = np.ptp(points[:, 0])
        c_len = np.ptp(points[:, 1])
        longest_dim = max(r_len, c_len)

        # 조건 검사 후, 분할
        if longest_dim > max_extent:
            try:
                # PCA를 이용한 초기화
                pca = PCA(n_components=1)
                pca.fit(points)
                
                direction = pca.components_[0]
                center = points.mean(axis=0) 
                std_dev = np.sqrt(pca.explained_variance_[0])
                
                # 주축의 양 끝점으로 K-means의 초기 중심점 설정
                init_centers = np.array([
                    center + direction * std_dev,
                    center - direction * std_dev
                ])
                
                # K-Means 이중분할 수행
                kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1)
                labels = kmeans.fit_predict(points)
                
                # 자식 클러스터 생성
                points_a = points[labels == 0]
                points_b = points[labels == 1]

                # 자식이 0개가 되는 예외 상황 처리 
                if len(points_a) == 0 or len(points_b) == 0:
                     return [self._get_medoid(points)]
                
                # 재귀 호출
                goals_a = self._recursive_split(points_a, max_extent, min_split_points)
                goals_b = self._recursive_split(points_b, max_extent, min_split_points)
                
                return goals_a + goals_b

            except Exception as e:
                # PCA나 K-Means에서 예외 발생 시 안전하게 현재 평균 반환
                return [self._get_medoid(points)]
        else:
            # 적당한 크기라면 현재 중심 반환
            return [self._get_medoid(points)]






