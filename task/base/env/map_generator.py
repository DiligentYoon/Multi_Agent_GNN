import numpy as np
from typing import Tuple, Optional, List
from abc import abstractmethod
import random
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

class MapBase:
    def __init__(self, cfg: dict, extra_info: dict):
        self.cfg = cfg
        self.res_m = cfg.get("resolution", 0.01)
        self.map_mask = cfg.get("map_representation", {
            "free": 0, "occupied": 1, "unknown": 2, "start": 3, "goal": 4
        })

        if all(key in extra_info for key in ['width', 'height']):
            self.meters_h = extra_info.get('height')
            self.meters_w = extra_info.get('width')
        else:
            raise RuntimeError("Omit Necessary Key [width, height]")

        self.H = int(round(self.meters_h / self.res_m))
        self.W = int(round(self.meters_w / self.res_m))

        self.gt = np.full((self.H, self.W), self.map_mask["free"], dtype=np.int8)
        self.belief = np.full((self.H, self.W), self.map_mask["unknown"], dtype=np.int8)
        self.belief_frontier = np.full((self.H, self.W), self.map_mask["unknown"], dtype=np.int8)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        H, W = self.H, self.W
        col = int(np.clip(x / self.res_m, 0, W - 1))
        row_from_bottom = int(np.clip(y / self.res_m, 0, H - 1))
        row = (H - 1) - row_from_bottom
        return row, col

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        H, W = self.H, self.W
        y_from_bottom = (H - 1 - row) * self.res_m
        x = col * self.res_m
        y = y_from_bottom
        return x, y

    def world_to_grid_np(self, world: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        if world.ndim == 1:
            world = world.reshape(1, -1)
        x = world[:, 0]
        y = world[:, 1]
        col = np.clip(x / self.res_m, 0, W - 1).astype(np.long)
        row = (H - 1) - (np.clip(y / self.res_m, 0, H - 1)).astype(np.long)

        grid_position = np.stack((col, row), axis=-1)

        return grid_position

    def grid_to_world_np(self, grid: np.ndarray) -> np.ndarray:
        H, W = self.H, self.W
        if grid.ndim == 1:
            grid = grid.reshape(1, -1)
        col = grid[:, 0]
        row = grid[:, 1]
        x = col * self.res_m
        y = (H - 1 - row) * self.res_m    

        world = np.stack((x, y), axis=-1)

        return world

    def initialization(self):
        self.gt.fill(self.map_mask["free"])
        self.belief.fill(self.map_mask["unknown"])
        self.belief_frontier.fill(self.map_mask["unknown"])

    def add_border_walls(self, thickness_m: float = 0.05):
        t = max(1, int(round(thickness_m / self.res_m)))
        self.gt[:t, :] = self.map_mask["occupied"]
        self.gt[-t:, :] = self.map_mask["occupied"]
        self.gt[:, :t] = self.map_mask["occupied"]
        self.gt[:, -t:] = self.map_mask["occupied"]

    def add_rect_obstacle(self, 
                          xmin: float, ymin: float, 
                          xmax: float, ymax: float):
        r1, c1 = self.world_to_grid(max(0.0, xmin), max(0.0, ymin))
        r2, c2 = self.world_to_grid(min(self.meters_w, xmax), min(self.meters_h, ymax))
        r_lo, r_hi = sorted((r1, r2))
        c_lo, c_hi = sorted((c1, c2))
        self.gt[r_lo:r_hi+1, c_lo:c_hi+1] = self.map_mask["occupied"]
    
    def add_random_obstacles(self, 
                             n: int = 5,
                             max_attempts: int =1000, 
                             min_m: float = 0.15, 
                             max_m: float = 0.3,
                             min_x_m: float = 0.25,
                             min_dist: float = 0.1,
                             seed: Optional[int] = None):
        """Place N rectangular obstacles of random size with min/max dimensions."""
        rng = np.random.default_rng(seed)
        count = 0
        attempts = 0
        while count < n and attempts < max_attempts:
            attempts += 1
            w = rng.uniform(min_m, max_m)
            h = rng.uniform(min_m, max_m)
            x = rng.uniform(min_x_m, max(min_x_m, self.meters_w - w))
            y = rng.uniform(0.0, max(min_x_m, self.meters_h - h))

            if self.check_validity(x, y, w, h, min_dist=min_dist):
                self.add_rect_obstacle(x, y, x+w, y+h)
                count += 1
            
            if attempts == max_attempts:
                print("Max Attempts")

    def check_validity(self, x, y, w, h, min_dist: float = 0.1):
            """
            생성하려는 장애물 영역(x, y, w, h)을 min_dist 만큼 확장하여 검사
            """
            pad_x_min = x - min_dist
            pad_y_min = y - min_dist
            pad_x_max = x + w + min_dist
            pad_y_max = y + h + min_dist

            r1, c1 = self.world_to_grid(pad_x_min, pad_y_min)
            r2, c2 = self.world_to_grid(pad_x_max, pad_y_max)

            r_min, r_max = sorted((r1, r2))
            c_min, c_max = sorted((c1, c2))

            roi = self.gt[r_min:r_max+1, c_min:c_max+1]
 
            has_start = np.any(roi == self.map_mask["start"])
            has_goal = np.any(roi == self.map_mask["goal"])
            
            has_obs = np.any(roi == self.map_mask["occupied"])

            if has_start or has_goal or has_obs:
                return False
            
            return True
    
    @abstractmethod
    def reset_gt_and_belief(self):
        raise NotImplementedError
    

class CorridorMap(MapBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg, cfg['corridor'])

    def add_start_and_goal_zones(self, wall_thickness_m=0.05, thickness_m=0.25):
        H, W = self.H, self.W
        wall_t = max(1, int(round(wall_thickness_m / self.res_m)))
        t = max(1, int(round(thickness_m / self.res_m)))
        self.gt[wall_t-1:H-wall_t, wall_t:wall_t+t+1] = self.map_mask["start"]
        self.gt[wall_t-1:H-wall_t, W-1-wall_t-t:W-wall_t] = self.map_mask["goal"]

    def reset_gt_and_belief(self, seed: int = None):
        self.initialization()
        self.add_border_walls()
        self.add_start_and_goal_zones()
        self.add_random_obstacles(n=10, min_m=0.15, max_m=0.3, seed=seed)


class MazeMap(MapBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg, cfg['maze'])

        self.corridor_width = cfg['maze'].get('corridor_width', 0.3)
        
        self.maze_rows = int(round(self.meters_h / self.corridor_width))
        self.maze_cols = int(round(self.meters_w / self.corridor_width))

    def _generate_maze_grid(self, seed):
            """Recursive Backtracking (Matrix Coordinates: Top-Left is (0,0))"""
            rng = np.random.default_rng(seed)
            maze = np.ones((self.maze_rows, self.maze_cols), dtype=int) 
            start_r, start_c = self.maze_rows - 1, 0
            maze[start_r, start_c] = 0
            stack = [(start_r, start_c)]
            
            # DFS Loop
            while stack:
                r, c = stack[-1]
                neighbors = []
                # 2칸씩 점프 (Matrix Index 기준 상하좌우)
                directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
                
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.maze_rows and 0 <= nc < self.maze_cols:
                        if maze[nr, nc] == 1: # Unvisited Wall
                            neighbors.append((nr, nc, dr, dc))
                
                if neighbors:
                    nr, nc, dr, dc = rng.choice(neighbors)
                    maze[nr, nc] = 0               # 타겟 셀 뚫기
                    maze[r + dr//2, c + dc//2] = 0 # 사이 벽 뚫기
                    stack.append((nr, nc))
                else:
                    stack.pop()
            goal_r, goal_c = 0, self.maze_cols - 1
            maze[goal_r, goal_c] = 0
            
            if goal_c > 0 and maze[goal_r, goal_c - 1] == 1:
                maze[goal_r, goal_c - 1] = 0 # 왼쪽 뚫기
            elif goal_r < self.maze_rows - 1 and maze[goal_r + 1, goal_c] == 1:
                maze[goal_r + 1, goal_c] = 0 # 아래쪽 뚫기

            for r in range(self.maze_rows):
                for c in range(self.maze_cols):
                    if maze[r, c] == 0: # Path
                        # r=0(Top) -> y=Max, r=Max(Bottom) -> y=0
                        r_inv = (self.maze_rows - 1) - r
                        
                        ymin = r_inv * self.corridor_width
                        xmin = c * self.corridor_width
                        ymax = ymin + self.corridor_width
                        xmax = xmin + self.corridor_width
                        
                        # World -> Grid Index (고해상도 맵)
                        r1, c1 = self.world_to_grid(xmin, ymin)
                        r2, c2 = self.world_to_grid(xmax, ymax)
                        r_lo, r_hi = sorted((r1, r2))
                        c_lo, c_hi = sorted((c1, c2))
                        
                        self.gt[r_lo:r_hi+1, c_lo:c_hi+1] = self.map_mask["free"]
    
    def add_start_and_goal_zones(self, thickness: float = 0.05):
            r_start = self.maze_rows - 1
            r_start_inv = (self.maze_rows - 1) - r_start
            
            s_ymin = r_start_inv * self.corridor_width
            s_xmin = 0.0 # First Col
            
            # 벽 두께만큼 띄우고(margin), 길 폭만큼 설정
            sy_min_real = s_ymin + thickness
            sx_min_real = s_xmin + thickness
            sy_max_real = s_ymin + self.corridor_width
            sx_max_real = s_xmin + self.corridor_width

            rs1, cs1 = self.world_to_grid(sx_min_real, sy_min_real)
            rs2, cs2 = self.world_to_grid(sx_max_real, sy_max_real)
            
            self.gt[min(rs1, rs2):max(rs1, rs2)+1, min(cs1, cs2):max(cs1, cs2)+1] = self.map_mask["start"]

            g_xmax_real = self.meters_w - thickness
            g_ymax_real = self.meters_h - thickness
            
            g_xmin_real = (self.maze_cols - 1) * self.corridor_width
            g_ymin_real = (self.maze_rows - 1) * self.corridor_width

            rg1, cg1 = self.world_to_grid(g_xmin_real, g_ymin_real)
            rg2, cg2 = self.world_to_grid(g_xmax_real, g_ymax_real)
            
            self.gt[min(rg1, rg2)+5:max(rg1, rg2)+4, min(cg1, cg2):max(cg1, cg2)] = self.map_mask["goal"]

    def reset_gt_and_belief(self, seed=None):
        self.initialization()
        # 벽으로 모두 채우고, 유효 Path만 뚫기
        self.gt.fill(self.map_mask["occupied"])
        self._generate_maze_grid(seed)
        self.add_border_walls()
        self.add_start_and_goal_zones()
        self.add_random_obstacles(n=10, min_m=0.05, max_m=0.1, seed=seed)


class RandomObstacleMap(MapBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg, cfg['random'])
    
    def add_start_and_goal_zones(self, goal_zone_size: float = 0.3, thickness: float = 0.05):
        sz = goal_zone_size
        s_xmin, s_ymin = thickness, thickness

        # Start
        rs1, cs1 = self.world_to_grid(s_xmin, s_ymin)
        rs2, cs2 = self.world_to_grid(s_xmin+sz, s_ymin+sz)
        self.gt[min(rs1, rs2):max(rs1, rs2)+1, min(cs1, cs2):max(cs1, cs2)+1] = self.map_mask["start"]

        # Goal 
        g_xmin, g_ymin = self.meters_w - sz - thickness, self.meters_h - sz - thickness
        rg1, cg1 = self.world_to_grid(g_xmin, g_ymin)
        rg2, cg2 = self.world_to_grid(g_xmin+sz, g_ymin+sz)
        self.gt[min(rg1, rg2)+1:max(rg1, rg2), min(cg1, cg2):max(cg1, cg2)] = self.map_mask["goal"]

    def reset_gt_and_belief(self, seed=None):
        self.initialization()
        self.add_border_walls()
        self.add_start_and_goal_zones()
        self.add_random_obstacles(n=15, min_m=0.2, max_m=0.3, seed=seed)


class SingleMazeMap(MapBase):
    def __init__(self, cfg: dict):
        super().__init__(cfg, cfg["single_maze"])
        
        self.corridor_width = cfg['single_maze'].get('corridor_width', 0.3)
        
        self.maze_rows = int(self.meters_h / self.corridor_width)
        self.maze_cols = int(self.meters_w / self.corridor_width)

    def _generate_path(self, seed, max_steps=None):
        """
        단일 경로(Non-Branching) 보장: Loop-Erased Random Walk (LERW)
        """
        rng = np.random.default_rng(seed)

        rows, cols = self.maze_rows, self.maze_cols
        start = (rows - 1, 0)
        goal  = (0, cols - 1)

        # 안전장치(필수는 아니지만 권장): 너무 오래 걸리면 리스타트
        if max_steps is None:
            max_steps = rows * cols * 200  # 경험적 상수 (원하면 조절)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def neighbors(cell):
            r, c = cell
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield (nr, nc)

        # LERW를 “성공할 때까지” 시도 (finite grid에서 goal hit는 확률 1이지만, 시간 제한을 둠)
        for attempt in range(1000):
            path = [start]
            index = {start: 0}  # cell -> position in path
            cur = start
            steps = 0

            while cur != goal and steps < max_steps:
                nbs = list(neighbors(cur))
                nxt = nbs[rng.integers(len(nbs))]

                if nxt in index:
                    # loop 발생: path를 nxt가 있던 지점까지 잘라냄
                    cut = index[nxt]
                    # 잘려나가는 구간의 index 엔트리 제거
                    for cell in path[cut+1:]:
                        index.pop(cell, None)
                    path = path[:cut+1]
                else:
                    path.append(nxt)
                    index[nxt] = len(path) - 1

                cur = path[-1]
                steps += 1

            if cur == goal:
                logic_map = np.zeros((rows, cols), dtype=int)
                for r, c in path:
                    logic_map[r, c] = 1
                return logic_map

        # 여기까지 오면(매우 드묾) 실패: 최소 안전하게 “직선+직선” 경로라도 반환
        logic_map = np.zeros((rows, cols), dtype=int)
        r, c = start
        logic_map[r, c] = 1
        # 위로 이동
        while r > goal[0]:
            r -= 1
            logic_map[r, c] = 1
        # 오른쪽 이동
        while c < goal[1]:
            c += 1
            logic_map[r, c] = 1
        return logic_map

    def _render_logic_to_gt(self, logic_map):
        """
        맵을 고해상도 GT 맵에 렌더링
        """
        self.gt.fill(self.map_mask["occupied"])

        eps = 1e-6
        H, W = self.gt.shape

        for r in range(self.maze_rows):
            for c in range(self.maze_cols):
                if logic_map[r, c] != 1:
                    continue

                # logic row(top-down) -> world y(bottom-up)
                r_inv = (self.maze_rows - 1) - r

                xmin = c * self.corridor_width
                xmax = (c + 1) * self.corridor_width
                ymin = r_inv * self.corridor_width
                ymax = (r_inv + 1) * self.corridor_width

                # 경계 좌표는 +/- eps로 내부를 찍어서 라운딩/경계 포함 문제를 피함
                r1, c1 = self.world_to_grid(xmin + eps, ymin + eps)
                r2, c2 = self.world_to_grid(xmax - eps, ymax - eps)

                r_lo, r_hi = sorted((r1, r2))
                c_lo, c_hi = sorted((c1, c2))

                # clamp
                r_lo = max(0, min(H - 1, r_lo))
                r_hi = max(0, min(H - 1, r_hi))
                c_lo = max(0, min(W - 1, c_lo))
                c_hi = max(0, min(W - 1, c_hi))

                # 상한 포함(+1)
                self.gt[r_lo:r_hi + 1, c_lo:c_hi + 1] = self.map_mask["free"]

    def add_start_and_goal_zones(self, thickness: float = 0.05):
            """Start/Goal 마킹 (MazeMap의 동기화된 로직 사용)"""
            
            r_start = self.maze_rows - 1
            r_start_inv = (self.maze_rows - 1) - r_start # r_inv=0
            
            s_ymin = r_start_inv * self.corridor_width
            s_xmin = 0.0
            
            sy_min_real = s_ymin + thickness
            sx_min_real = s_xmin + thickness
            sy_max_real = s_ymin + self.corridor_width
            sx_max_real = s_xmin + self.corridor_width

            rs1, cs1 = self.world_to_grid(sx_min_real, sy_min_real)
            rs2, cs2 = self.world_to_grid(sx_max_real, sy_max_real)
            
            self.gt[min(rs1, rs2):max(rs1, rs2)+1, min(cs1, cs2):max(cs1, cs2)+1] = self.map_mask["start"]

            r_goal = 0
            r_goal_inv = (self.maze_rows - 1) - r_goal
            
            g_xmin_real = (self.maze_cols - 1) * self.corridor_width
            g_ymin_real = r_goal_inv * self.corridor_width 
            
            g_xmax_real = self.meters_w - thickness
            g_ymax_real = self.meters_h - thickness

            rg1, cg1 = self.world_to_grid(g_xmin_real, g_ymin_real)
            rg2, cg2 = self.world_to_grid(g_xmax_real, g_ymax_real)
            
            self.gt[min(rg1, rg2):max(rg1, rg2), min(cg1, cg2):max(cg1, cg2)] = self.map_mask["goal"]

    def reset_gt_and_belief(self, seed=None):
        self.initialization()
        logic_map = self._generate_path(seed)
        self._render_logic_to_gt(logic_map)
        self.add_border_walls()
        self.add_start_and_goal_zones()
        self.add_random_obstacles(n=10, min_m=0.03, max_m=0.1, seed=seed, max_attempts=10000, min_dist=0.1)


def visualize_maps(maps_list, save_path="generated_maps.png", flip_rows=True):
    """
    생성된 맵 리스트를 받아 Matplotlib으로 시각화하고 저장하는 함수
    - x 라벨 겹침 방지: 맨 아래 행에만 x-label/tick 표시
    - flip_rows=True이면 이미지 행을 상하 반전(np.flipud)하여 표시
    """
    n_maps = len(maps_list)
    fig, axes = plt.subplots(n_maps, 2, figsize=(12, 5 * n_maps))
    axes = np.atleast_2d(axes)  # n_maps=1이어도 (1,2) 형태로 정규화

    # 0:Free, 1:Occupied, 2:Unknown, 3:Start, 4:Goal
    cmap = mcolors.ListedColormap(['white', 'black', 'lightgray', 'limegreen', 'red'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im_gt = None  # colorbar용 핸들

    for i, (map_name, map_obj) in enumerate(maps_list):
        gt = map_obj.gt
        bl = map_obj.belief

        # 행(세로) 뒤집기
        if flip_rows:
            gt = np.flipud(gt)
            bl = np.flipud(bl)

        # Ground Truth Plot
        ax_gt = axes[i, 0]
        im_gt = ax_gt.imshow(gt, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        ax_gt.set_title(f"{map_name} - Ground Truth")
        ax_gt.set_ylabel("Grid Y")

        # Belief Plot
        ax_bl = axes[i, 1]
        ax_bl.imshow(bl, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        ax_bl.set_title(f"{map_name} - Belief (Initial)")
        ax_bl.set_ylabel("Grid Y")

        # x 라벨/틱 겹침 방지: 마지막 행에만 표시
        if i == n_maps - 1:
            ax_gt.set_xlabel("Grid X")
            ax_bl.set_xlabel("Grid X")
        else:
            ax_gt.set_xlabel("")
            ax_bl.set_xlabel("")
            ax_gt.tick_params(axis='x', labelbottom=False)
            ax_bl.tick_params(axis='x', labelbottom=False)

    # colorbar (기존 방식 유지)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(['Free', 'Occupied', 'Unknown', 'Start', 'Goal'])

    # 레이아웃 여유(라벨 겹침 추가 완화)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.subplots_adjust(hspace=0.35)  # 필요 시 수치 조정

    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"이미지가 저장되었습니다: {save_path}")


def save_each_map_gt(maps_list,
                     out_dir="maps_out",
                     flip_rows=True,
                     dpi=300,
                     seed=0):
    os.makedirs(out_dir, exist_ok=True)

    # 0:Free, 1:Occupied, 2:Unknown, 3:Start, 4:Goal
    # cmap = mcolors.ListedColormap(['white', 'black', 'lightgray', 'limegreen', 'red'])
    color_255 = np.array([
        [255, 255, 255],
        [0, 0, 0],
        [230, 230, 230],
        [50, 200, 80],
        [180, 50, 200],
    ], dtype=float)
    cmap = mcolors.ListedColormap(color_255 / 255.0)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    for map_name, map_obj in maps_list:
        gt = map_obj.gt
        if flip_rows:
            gt = np.flipud(gt)

        # (선택) 셀 해상도 그대로 저장하고 싶으면 figsize를 배열 크기에 맞춤
        H, W = gt.shape
        fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  # 여백 0
        ax.imshow(gt, cmap=cmap, norm=norm, origin='lower', interpolation='nearest')
        ax.set_axis_off()

        safe_name = map_name.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(out_dir, f"{safe_name}_{seed}.png")
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        print(f"Saved: {save_path}")


def main():
    seeds = [165, 166, 167, 168, 169, 170, 171, 172, 173, 174]

    for seed in seeds:
    
        # (A) Corridor Map (1m x 5m)
        cfg_corridor = {
            "height": 1.0, 
            "width": 5.0, 
            "resolution": 0.01,
            "map_representation": {"free": 0, "occupied": 1, "unknown": 2, "start": 3, "goal": 4}
        }
        cfg_corridor = {"corridor": cfg_corridor}
        corridor = CorridorMap(cfg_corridor)
        corridor.reset_gt_and_belief(seed=seed)

        # (B) Maze Map (2.5m x 2.5m)
        cfg_maze = {
            "height": 2.5, 
            "width": 2.5, 
            "resolution": 0.01,
            "corridor_width": 0.3, # 미로 길 폭 25cm
            "map_representation": {"free": 0, "occupied": 1, "unknown": 2, "start": 3, "goal": 4}
        }
        cfg_maze = {"maze": cfg_maze}
        maze = MazeMap(cfg_maze)
        maze.reset_gt_and_belief(seed=seed)

        # (C) Random Obstacle Map (2.5m x 2.5m)
        cfg_random = {
            "height": 2.5, 
            "width": 2.5, 
            "resolution": 0.01,
            "map_representation": {"free": 0, "occupied": 1, "unknown": 2, "start": 3, "goal": 4}
        }
        cfg_random = {"random": cfg_random}
        random_map = RandomObstacleMap(cfg_random)
        random_map.reset_gt_and_belief(seed=seed)


        # (D) Single Maze Map (2.5m x 2.5m)
        cfg_single_maze = {
            "height": 2.5, 
            "width": 2.5, 
            "resolution": 0.01,
            "corridor_width": 0.3, # 미로 길 폭 25cm
            "map_representation": {"free": 0, "occupied": 1, "unknown": 2, "start": 3, "goal": 4}
        }
        cfg_single_maze = {"single_maze": cfg_single_maze}
        single_maze = SingleMazeMap(cfg_single_maze)
        single_maze.reset_gt_and_belief(seed=seed)
        
        # 시각화 실행
        maps_to_plot = [
            ("Corridor Map", corridor),
            ("Maze Map", maze),
            ("Random Obstacle Map", random_map),
            ("Single Maze Map", single_maze)
        ]

        save_each_map_gt(maps_to_plot, out_dir='gt_maps', flip_rows=True, dpi=300, seed=seed)
        # visualize_maps(maps_to_plot, save_path="test_maps_output.png")

if __name__ == "__main__":
    main()