import numpy as np
import heapq
from scipy.ndimage import binary_dilation

from ..base.env.env import MapInfo

class AStarNode:
    """
    A* 알고리즘에 사용될 노드 객체
    """
    def __init__(self, position, parent=None):
        self.position = position  # (row, col) 튜플
        self.parent = parent
        self.g = 0  # 시작 노드로부터의 비용
        self.h = 0  # 목표 노드까지의 추정 비용 (Heuristic)
        self.f = 0  # 총 비용 (g + h)

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


def astar_search(map_info: MapInfo, 
                 start_pos: np.ndarray | tuple, 
                 end_pos: np.ndarray | tuple,
                 agent_id: int,
                 inflation_radius_cells: int = 3) -> np.ndarray | None:
    """
    A* 알고리즘 기반 최적경로 탐색
    """
    # 1. 입력 위치를 튜플로 일관성 있게 변환
    if isinstance(start_pos, np.ndarray):
        start_pos = tuple(start_pos.flatten())
    if isinstance(end_pos, np.ndarray):
        end_pos = tuple(end_pos.flatten())

    H, W = map_info.H, map_info.W
        
    inflated_map = inflate_obstacles(map_info, inflation_radius_cells=inflation_radius_cells)
    map_mask = map_info.map_mask

    # Start Pos 예외처리 
    if inflated_map[start_pos] == map_mask["occupied"]:
        # print(f"[INFO] Agent {agent_id}: Start Pose is occupied")
        start_min_r = max(0, start_pos[0] - inflation_radius_cells*2)
        start_max_r = min(H-1, start_pos[0] + inflation_radius_cells*2)
        start_min_c = max(0, start_pos[1] - inflation_radius_cells*2)
        start_max_c = min(W-1, start_pos[1] + inflation_radius_cells*2)

        valid = inflated_map[start_min_r:start_max_r, start_min_c:start_max_c] == map_mask["free"]
        if np.any(valid):
            valid_cells = np.column_stack(np.where(valid)) + np.array([start_min_r, start_min_c])
            min_cell_id = np.argmin(np.linalg.norm(valid_cells - start_pos, axis=1))
            start_pos = (valid_cells[min_cell_id, 0], valid_cells[min_cell_id, 1])
        else:
            # print(f"[INFO] Agent {agent_id}: Inflated Start Pose is occupied too")
            return None
    
    # End Pos 예외처리
    if inflated_map[end_pos] == map_mask["occupied"]:
        # print(f"[INFO] Agent {agent_id}: End Pose is occupied")
        end_min_r = max(0, end_pos[0] - inflation_radius_cells*2)
        end_max_r = min(H-1, end_pos[0] + inflation_radius_cells*2)
        end_min_c = max(0, end_pos[1] - inflation_radius_cells*2)
        end_max_c = min(W-1, end_pos[1] + inflation_radius_cells*2)

        valid = inflated_map[end_min_r:end_max_r, end_min_c:end_max_c] == map_mask["free"]
        if np.any(valid):
            valid_cells = np.column_stack(np.where(valid)) + np.array([end_min_r, end_min_c])
            min_cell_id = np.argmin(np.linalg.norm(valid_cells - end_pos, axis=1))
            end_pos = (valid_cells[min_cell_id, 0], valid_cells[min_cell_id, 1])
        else:
            # print(f"[INFO] Agent {agent_id}: Inflated End Pose is occupied too")
            return None
    
    # 시작/끝 노드 및 맵 정보 초기화
    start_node = AStarNode(start_pos)
    end_node = AStarNode(end_pos)

    # 2. open_list 및 closed_list 초기화
    open_list = []  # 최소 힙 (Priority Queue)
    heapq.heappush(open_list, start_node)
    
    open_set_lookup = {start_node.position: start_node}
    
    closed_set = set() # 방문 완료한 노드 위치 저장

    grid_rows, grid_cols = inflated_map.shape

    while open_list:
        current_node = heapq.heappop(open_list)
        if current_node.position in closed_set:
            continue
        open_set_lookup.pop(current_node.position)
        closed_set.add(current_node.position)

        # 목표 도달 시 경로 역추적
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return np.array(path[::-1], dtype=np.int32)

        # 8방향 이웃 탐색
        for move in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor_pos = (current_node.position[0] + move[0], current_node.position[1] + move[1])

            # 그리드 범위 및 방문 여부 확인
            if not (0 <= neighbor_pos[0] < grid_rows and 0 <= neighbor_pos[1] < grid_cols):
                continue
            if neighbor_pos in closed_set:
                continue
            if inflated_map[neighbor_pos] == map_mask["occupied"]:
                continue

            # 3. 비용 계산 및 노드 업데이트
            move_cost = 1.414 if abs(move[0]) == 1 and abs(move[1]) == 1 else 1.0
            g_cost = current_node.g + move_cost
            
            if neighbor_pos not in open_set_lookup or g_cost < open_set_lookup[neighbor_pos].g:
                dx = neighbor_pos[0] - end_node.position[0]
                dy = neighbor_pos[1] - end_node.position[1]
                h_cost = np.sqrt(dx*dx + dy*dy)
                
                neighbor_node = AStarNode(neighbor_pos, current_node)
                neighbor_node.g = g_cost
                neighbor_node.h = h_cost
                neighbor_node.f = g_cost + h_cost
                
                heapq.heappush(open_list, neighbor_node)
                open_set_lookup[neighbor_pos] = neighbor_node

    # print(f"[INFO] Agent {agent_id}: No path found")
    return None


def inflate_obstacles(map_info: MapInfo, inflation_radius_cells: int = 2) -> np.ndarray:
    """
    Belief map의 장애물 팽창
    """
    belief_map = map_info.belief
    map_mask = map_info.map_mask
    
    if inflation_radius_cells <= 0:
        return np.copy(belief_map)
        
    # 1. 장애물만 1로 표시된 이진 맵 생성
    obstacle_mask = (belief_map == map_mask["occupied"])

    # 2. 팽창에 사용할 구조 요소(커널) 생성
    # inflation_radius_cells가 5이면 11x11 크기의 정사각형 커널
    structure_size = 2 * inflation_radius_cells + 1
    structure = np.ones((structure_size, structure_size))
    
    # 3. Scipy의 binary_dilation 함수를 사용하여 팽창 연산 수행
    dilated_obstacle_mask = binary_dilation(obstacle_mask, structure=structure)
    
    # 4. 원본 맵에 팽창된 장애물 영역을 덮어쓰기
    inflated_map = np.copy(belief_map)
    inflated_map[dilated_obstacle_mask] = map_mask["occupied"]
    
    return inflated_map


def is_path_valid(map_info: MapInfo, path_cells: np.ndarray) -> bool:
    """
    주어진 경로가 현재 belief map 상에서 유효한지(장애물과 충돌하지 않는지) 확인합니다.
    """
    if path_cells is None or len(path_cells) == 0:
        return False
    
    rows, cols = path_cells[:, 0], path_cells[:, 1]
    
    # 경로가 맵 범위를 벗어나는지 확인
    H, W = map_info.H, map_info.W
    if np.any(rows < 0) or np.any(rows >= H) or np.any(cols < 0) or np.any(cols >= W):
        return False

    # belief map을 기준으로 경로상의 장애물 충돌 여부 확인
    path_values = map_info.belief[rows, cols]
    if np.any(path_values == map_info.map_mask["occupied"]):
        return False
        
    return True