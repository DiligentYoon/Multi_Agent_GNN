import numpy as np

def split_and_score_local_region(
    local_cell_bounds, agents_pose, map_info, 
    S_min=8, max_local=16,     
    alpha=1.0, beta=0.2, gamma=0.1, goal_w=10.0):
    """
    하나의 '유효 영역'을 받아, 
    더 세밀하게 분할(Local KD-Tree)하여
    최종 타겟 영역들의 목록 및 점수를 반환
    
    """
    belief = map_info.belief
    map_mask = map_info.map_mask
    H, W = map_info.H, map_info.W # 전체 맵 크기
    
    # --- stats_local 함수 ---
    def stats_local(r0, r1, c0, c1):
        sub = belief[r0:r1, c0:c1]
        area = max(1, (r1-r0)*(c1-c0))
        F = int(np.sum(sub == map_mask["frontier"]))
        f = F / area
        h, w = (r1-r0), (c1-c0)
        return f, h, w

    r0, r1, c0, c1 = local_cell_bounds
    # 경계 검사 (필요시)
    r0 = max(0, min(r0, H)); r1 = max(0, min(r1, H))
    c0 = max(0, min(c0, W)); c1 = max(0, min(c1, W))

    kept_regions = [] # 최종 반환될 하위 영역 목록
    kept_scores = []
    stack = [(r0, r1, c0, c1)]

    while stack:
        # 최대 개수 도달 시 중단
        if len(kept_regions) >= max_local:
            break
            
        r0, r1, c0, c1 = stack.pop()
        f, h, w = stats_local(r0, r1, c0, c1)

        # 중단 조건
        small_enough = (h <= S_min) or (w <= S_min)
        
        stop = small_enough

        if stop:
            h, w = (r1-r0), (c1-c0)
            area = max(1, h*w)

            unk = np.sum(belief[r0:r1, c0:c1] == map_mask["unknown"]) / area
            occ = np.sum(belief[r0:r1, c0:c1] == map_mask["occupied"]) / area
            free = np.sum(belief[r0:r1, c0:c1] == map_mask["free"]) / area
            goal = np.sum(belief[r0:r1, c0:c1] == map_mask["goal"]) / area


            r = (r0 + r1) / 2
            c = (c0 + c1) / 2
            cx, cy = map_info.grid_to_world(r, c)

            d_list = [float(np.hypot(cx - ax, cy - ay)) for (ax, ay) in agents_pose]
            d_norm = np.mean(d_list)

            J_now = alpha*unk - beta*(occ+free) - gamma * d_norm + goal_w*goal
        
            kept_regions.append((r0, r1, c0, c1))
            kept_scores.append(J_now)
            continue

        # KD 분할 (긴 축 기준)
        if h >= w:
            rm = (r0 + r1) // 2
            stack.append((r0, rm, c0, c1))
            stack.append((rm, r1, c0, c1))
        else:
            cm = (c0 + c1) // 2
            stack.append((r0, r1, c0, cm))
            stack.append((r0, r1, cm, c1))
    
    return kept_regions, kept_scores