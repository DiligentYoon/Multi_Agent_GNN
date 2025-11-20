import copy
import numpy as np
import torch

from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt as edt
from torch_geometric.data import Data, Batch

def get_nominal_control(p_target: list[np.ndarray] | np.ndarray,
                        on_search: list[bool] | np.ndarray,
                        v_current: list[float] | np.ndarray,
                        a_max: float,
                        w_max: float,
                        v_max: float,
                        k_v: float = 1.0,
                        k_w: float = 1.5) -> np.ndarray:

        if isinstance(p_target, list):
            p_target = np.vstack(p_target) # (n, 2)
        if isinstance(v_current, list):
            v_current = np.array(v_current).reshape(-1, 1) # (n, 1)
        if isinstance(on_search, list):
            on_search = np.array(on_search)
            k_v_arr = np.where(on_search, k_v*1.5, k_v)
            k_w_arr = np.where(on_search, k_w*1.5, k_w)

        lx, ly = p_target[:, 0], p_target[:, 1]
            
        dist_to_target = np.sqrt(lx**2 + ly**2)
        angle_to_target = np.arctan2(ly, lx)

        # Target velocity based on distance
        v_target = np.clip(k_v_arr * dist_to_target, 0.0, v_max).reshape(-1, 1)
        
        # P-control for acceleration
        a_ref = k_v * (v_target - v_current)
        a_ref = np.clip(a_ref, -a_max, a_max)

        # P-control for angular velocity
        w_ref = np.clip(k_w_arr * angle_to_target, -w_max, w_max).reshape(-1, 1)
        
        return np.hstack([a_ref, w_ref])


def obs_to_graph(obs: dict | list[dict], device: torch.device) -> Data:
    """
    """
    if isinstance(obs, dict):
        graph_features_np = obs['graph_features']
        edge_index_np = obs['edge_index']
        # PyTorch 텐서로 변환
        x = torch.tensor(graph_features_np, dtype=torch.float)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # 전체 관측을 나타내는 단일 Data 객체를 직접 생성
        data = Data(x=x, edge_index=edge_index)
    elif isinstance(obs, list):
        # Data 객체 리스트 생성
        data_list = []

        # 이중 리스트 구조 (에이전트 별 로컬 관측)
        if isinstance(obs[0], list):
            obs = [agent_obs for batch_obs in obs for agent_obs in batch_obs]

        for obs_dict in obs:
            x = torch.tensor(obs_dict['graph_features'], dtype=torch.float)
            edge_index = torch.tensor(obs_dict['edge_index'], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
        # Batch.from_data_list를 사용하여 하나의 배치 객체로 통합
        data = Batch.from_data_list(data_list)

    return data.to(device)


def obs_to_multi_graph(obs: dict | list[dict], device: torch.device) -> Data:
    """
    """
    if isinstance(obs, dict):
        graph_features_np = obs['graph_features']
        edge_index_np = obs['edge_index']
        # PyTorch 텐서로 변환
        x = torch.tensor(graph_features_np, dtype=torch.float)
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
        # 전체 관측을 나타내는 단일 Data 객체를 직접 생성
        data = Data(x=x, edge_index=edge_index)
    elif isinstance(obs, list):
        # Data 객체 리스트 생성
        data_list = []

        # 이중 리스트 구조 (에이전트 별 로컬 관측)
        if isinstance(obs[0], list):
            obs = [agent_obs for batch_obs in obs for agent_obs in batch_obs]

        for obs_dict in obs:
            x = torch.tensor(obs_dict['graph_features'], dtype=torch.float)
            edge_index = torch.tensor(obs_dict['edge_index'], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index))
        # Batch.from_data_list를 사용하여 하나의 배치 객체로 통합
        data = Batch.from_data_list(data_list)

    return data.to(device)


def create_fully_connected_edges(num_nodes: int) -> np.ndarray:
    """
    """
    if num_nodes <= 1:
        return np.empty((2, 0), dtype=np.int64)
    adj = ~np.eye(num_nodes, dtype=bool)
    edge_index = np.array(np.where(adj), dtype=np.int64)
    return edge_index


def world_to_local(w1: np.ndarray = None, w2: np.ndarray = None, yaw: float = None) -> np.ndarray:
    """
    Transforms a point from world coordinates to the robot's local frame.
    """
    if w1 is None:
        delta = w2
    else:
        delta = w2 - w1
    rot_mat = np.array([[np.cos(-yaw), -np.sin(-yaw)], 
                        [np.sin(-yaw),  np.cos(-yaw)]])
    
    local_pos = np.matmul(rot_mat, delta.transpose())
    return local_pos.transpose()


def local_to_world(w1:np.ndarray, l1: np.ndarray, yaw: float) -> np.ndarray:
    """
    Transforms a point from local coordinates to the world frame
        Inputs :
            wl  : World Frame Reference Point
            l1  : Local Frame Target Point
            yaw : Robot's Yaw Angle (rad) 
    """
    rot_mat = np.array([[np.cos(yaw), -np.sin(yaw)], 
                        [np.sin(yaw),  np.cos(yaw)]])
    
    world_pos = w1 + np.matmul(rot_mat, l1.transpose()).transpose()

    return world_pos


def collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask):
    """
    Ray-cast from (x0,y0) to (x1,y1) in cell coordinates.
        Inputs :

    """
    # 1) 정수 셀 인덱스로 변환
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))

    # 2) Bresenham 준비
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    # 3) 레이 캐스팅 루프
    while True:
        # 3.1) 맵 범위 체크
        if not (0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]):
            break

        # 3.2) 셀 클래스 읽기
        gt = ground_truth[y, x]

        if gt == map_mask["occupied"]:
            # 충돌 지점만 OCCUPIED로 업데이트하고 종료
            robot_belief[y, x] = map_mask["occupied"]
            break
        elif gt == map_mask["goal"]:
            robot_belief[y, x] = map_mask["goal"]
        else:
            # FREE 또는 기타(UNKNOWN) 영역은 FREE로 업데이트
            robot_belief[y, x] = map_mask["free"]

        # 3.3) 종료 조건: 끝점 도달
        if x == x1 and y == y1:
            break

        # 3.4) Bresenham step
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x   += sx
        if e2 <  dx:
            err += dx
            y   += sy

    return robot_belief

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's line algorithm
    """
    x0, y0 = int(round(x0)), int(round(y0))
    x1, y1 = int(round(x1)), int(round(y1))
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy

    while True:
        yield (y, x)  # (row, col) 순서로 반환

        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def normalize_angle(angle):
    """
    Normalize an angle to be within [0, 360) degrees.
    """
    return angle % 360


def calculate_fov_boundaries(center_angle, fov):
    """
    Calculate the start and end angles of the field of vision (FOV).
    
        Inputs:
            center_angle (float): The central angle of the FOV in degrees.
            fov (float): The total field of vision in degrees.
            
        Returns:
            (float, float): The start and end angles of the FOV.
    """
    half_fov = fov / 2
    
    start_angle = center_angle - half_fov
    end_angle = center_angle + half_fov
    
    start_angle = normalize_angle(start_angle)
    end_angle = normalize_angle(end_angle)
    
    return start_angle, end_angle

def fov_sweep(start_angle: float, end_angle: float, increment: int) ->np.ndarray:
    """
    Generate the correct sequence of angles to sweep the FOV from start to end with a specified increment.
    
        Inputs:
            start_angle (float): The starting angle of the FOV in degrees.
            end_angle (float): The ending angle of the FOV in degrees.
            increment (float): The angle increment in degrees.
            
        Returns:
            list: The sequence of angles representing the FOV sweep.
    """
    angles = []
    
    if start_angle < end_angle:
        angles = list(np.arange(start_angle, end_angle + increment, increment))
    else:
        angles = list(np.arange(start_angle, 360, increment)) + list(np.arange(0, end_angle + increment, increment))
    
    angles = [angle % 360 for angle in angles]
    
    angles_in_radians = np.radians(angles)

    return angles_in_radians

def sensor_work_heading(robot_position, 
                        sensor_range, 
                        robot_belief, 
                        ground_truth, 
                        heading, 
                        fov,
                        map_mask):
    """
    """

    sensor_angle_inc = 2.0
    if robot_position.shape[0] == 1:
        robot_position = robot_position.reshape(-1)
    x0 = robot_position[0]
    y0 = robot_position[1]
    start_angle, end_angle = calculate_fov_boundaries(heading, fov)
    sweep_angles = fov_sweep(start_angle, end_angle, sensor_angle_inc)

    x1_values = []
    y1_values = []
    
    for angle in sweep_angles:
        x1 = x0 + np.cos(angle) * sensor_range    
        y1 = y0 + np.sin(-angle) * sensor_range
        x1_values.append(x1)
        y1_values.append(y1)    
        
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief, map_mask)

    return robot_belief


def global_frontier_marking(map_info, reset_flag, frontier_cells: list[np.ndarray] = None):
    """
    """
    map = map_info
    belief = map.belief
    H, W = belief.shape
    UNKNOWN = map.map_mask["unknown"]
    FREE    = map.map_mask["free"]
    FRONTIER= map.map_mask["frontier"]

    # if reset_flag:
    # 전체 맵에 대한 Frontier 추출 및 마킹
    frontier_belief = copy.deepcopy(belief)
    free_mask    = (belief == FREE)
    unknown_mask = (belief == UNKNOWN)

    # 8-direction 이웃 Unknown 검사
    pad = np.pad(unknown_mask, 1, constant_values=False)
    unk_n8 = (
        pad[0:H,0:W]   + pad[0:H,1:W+1]   + pad[0:H,2:W+2] +
        pad[1:H+1,0:W] +                    pad[1:H+1,2:W+2] +
        pad[2:H+2,0:W] + pad[2:H+2,1:W+1] + pad[2:H+2,2:W+2]
    ).astype(np.uint8)

    # Free Cell -> 8-direction 중, unknown 존재 시, Frontier
    frontier_mask = free_mask & (unk_n8 > 0)
    rs, cs = np.where(frontier_mask)
    if rs.size:
        frontier_belief[rs, cs] = FRONTIER
    else:
        raise ValueError("Belief Map Initialization is failed")
    # else:
    #     # 시뮬레이션 진행 시, 이전 Frontier 맵에 대한 Incremental Update
    #     #   1. 이전 상태 유지
    #     #   2. 새로운 상태 업데이트
    #     frontier_belief = map.belief_frontier
    #     prev_frontier_rc = np.where(frontier_belief == FRONTIER)

    #     # 이번 스텝에서도 Frontier를 유지할 수 있는지 check
    #     r_prev, c_prev = prev_frontier_rc
    #     unknown_mask = (belief == UNKNOWN)
    #     free_center  = (belief[r_prev, c_prev] == FREE)

    #     pad = np.pad(unknown_mask, 1, constant_values=False)
    #     rp = r_prev + 1
    #     cp = c_prev + 1
    #     # 8-direction 이웃 Unknown 검사
    #     unk_n8 = (
    #         pad[rp-1, cp-1] + pad[rp-1, cp] + pad[rp-1, cp+1] +
    #         pad[rp,   cp-1]                  + pad[rp,   cp+1] +
    #         pad[rp+1, cp-1] + pad[rp+1, cp] + pad[rp+1, cp+1]
    #     )
    #     # Frontier 상태 유지 여부
    #     keep_mask = free_center & (unk_n8 > 0)
    #     remove_mask = ~keep_mask
    #     r_rm, c_rm = r_prev[remove_mask], c_prev[remove_mask]

    #     # 새로운 Frontier 상태
    #     flat = [arr for agent_list in frontier_cells for arr in agent_list if len(arr)>0]
    #     all_new = np.unique(np.concatenate(flat, axis=0), axis=0)
    #     r_new, c_new = all_new[:, 0], all_new[:, 1]

    #     # 상태 업데이트
    #     frontier_belief[r_rm, c_rm] = belief[r_rm, c_rm]
    #     frontier_belief[r_new, c_new] = FRONTIER

    return frontier_belief