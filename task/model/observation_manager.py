import numpy as np
import torch
import torch.nn as nn

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
        self.global_map_w = cfg["global_map_w"]
        self.global_map_h = cfg["global_map_h"]
        self.local_map_w = cfg["local_map_w"]
        self.local_map_h = cfg["local_map_h"]
        self.pooling_downsampling = cfg["pooling_downsampling"]
        self.device = device

        # Initializing full map, down-scaled map, info
        # obs/frontier/all pos/all trajectory/explored/explorable/history pos/history goal
        self.global_map = torch.zeros(8, self.global_map_h, self.global_map_w).float().to(device)
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
        H, W = self.global_map_h, self.global_map_w
        if world.ndim == 1:
            world = world.reshape(1, -1)
        x = world[:, 0]
        y = world[:, 1]
        col = np.clip(x / self.unit_size_m, 0, W - 1).astype(np.long)
        row = (H - 1) - (np.clip(y / self.unit_size_m, 0, H - 1)).astype(np.long)

        grid_position = np.stack((col, row), axis=-1)

        return grid_position


    def grid_to_world_np(self, grid: np.ndarray) -> np.ndarray:
        H, W = self.global_map_h, self.global_map_w
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
                                                                  (self.global_map_w, self.global_map_h)) # [col, row]
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
        self.global_map[0, :, :] = torch.from_numpy(obstacle).float()
        self.global_map[1, :, :] = torch.from_numpy(frontier).float()
        self.global_map[4, :, :] = torch.from_numpy(explored).float()
        self.global_map[5, :, :] = torch.from_numpy(explorable).float()
        self.global_map[2, :, :].fill_(0.)

        lmb = self.local_map_boundary # [col, row]
        self.global_pose = robot_pos # [x, y]
        agent_location = self.world_to_grid_np(self.global_pose) # [col, row]
        for e in range(self.num_robots):
            agent_location_r, agent_location_c = agent_location[e, 1], agent_location[e, 0]
            lmb[e] = get_local_map_boundaries((agent_location_r, agent_location_c), 
                                              (self.local_map_w, self.local_map_h), 
                                              (self.global_map_w, self.global_map_h)) # [col, row]
            agent_location_r = max(0, min(self.global_map_h, agent_location_r))
            agent_location_c = max(0, min(self.global_map_w, agent_location_c))
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
        global_input[6, :, :] -= global_input[2, :, :]
        global_input[7, :, :] = g_history
        dist_input = torch.zeros((self.num_robots, self.global_map_h, self.global_map_w))
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
        dist_input[self.global_map[1:2, :, :].repeat(self.num_robots, 1, 1) == 0] = 40
        for i in range(self.num_robots):
            agent_cell_pos = self.world_to_grid_np(self.global_pose[i, :2]).reshape(-1)
            dist_input[i, agent_cell_pos[1], agent_cell_pos[0]] = 40
        dist_input = -nn.MaxPool2d(self.pooling_downsampling)(-dist_input)
        dist_input[dist_input > 40] = 40
        global_input = torch.cat((global_input, dist_input), dim=0) # dim: [8 + Num_Agent, H, W]

        return global_input