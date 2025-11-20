import numpy as np

class Node:
    def __init__(self, bounds):
        self.bounds = bounds  # (r0, r1, c0, c1) - 이 노드가 담당하는 영역
        self.left = None      # 왼쪽 / 위쪽 자식 노드
        self.right = None     # 오른쪽 / 아래쪽 자식 노드
        self.axis = None      # 분할 축 (0: row(H), 1: col(W)) for 트리 탐색
        self.value = None     # 분할 값 for 트리 탐색
        self.information_gain = 0 # Node 고유 비용
        self.has_frontier = False
        
    def is_leaf(self):
        return self.left is None and self.right is None

    def __repr__(self):
        r0, r1, c0, c1 = self.bounds
        h, w = (r1 - r0), (c1 - c0)
        return f"Node(bounds={self.bounds}, size={h}x{w}, is_leaf={self.is_leaf()})"


class RegionKDTree:
    def __init__(self, bounds, valid_threshold=0.1, S_min=32):
        """
            Args: 
                bounds: 전체 맵의 경계 (r0, r1, c0, c1)
                S_min : 분할을 멈추는 최소 축 길이
        """
        self.S_min = S_min
        self.valid_threshold = valid_threshold
        self.root = None     # 트리의 루트 노드
        self.leaves = []     # 최종 분할된 영역(리프 노드) 리스트
        
        # KD-Tree 빌드
        self.root = self._build_recursive(bounds)

    def _build_recursive(self, bounds):
        """
            KD-Tree 빌드 함수
            Time Complexity : O(n) = nlog(n)
        """
        r0, r1, c0, c1 = bounds
        h = r1 - r0
        w = c1 - c0

        # 1. 새 노드 생성
        node = Node(bounds)

        # 2. 종료 조건: S_min
        # 높이나 너비 둘 중 하나라도 S_min보다 작거나 같으면 분할을 멈춤.
        if (h <= self.S_min) or (w <= self.S_min):
            self.leaves.append(node) # 리프 노드 리스트에 추가
            return node # 자식이 없는 리프 노드 반환

        # 3. KD 분할: 긴 축을 이등분
        if h >= w:
            # Row(height)가 더 길거나 같으므로, row를 분할 (수평 분할)
            rm = (r0 + r1) // 2
            
            node.axis = 0      # 0번 축(row)
            node.value = rm    # 분할 값
            node.left = self._build_recursive((r0, rm, c0, c1))
            node.right = self._build_recursive((rm, r1, c0, c1))
        else:
            # Column(width)이 더 기므로, column을 분할 (수직 분할)
            cm = (c0 + c1) // 2
            
            node.axis = 1      # 1번 축(col)
            node.value = cm    # 분할 값
            node.left = self._build_recursive((r0, r1, c0, cm))
            node.right = self._build_recursive((r0, r1, cm, c1))

        return node

    def update_node_states(self, map_info, agents_pose,
                           alpha=1.0, beta=0.2, gamma=0.5, goal_w=10.0):
        """
            Frontier가 속한 Valid Region들에 대해서 Information Gain 업데이트
            Inputs:
                map_info: MapInfo object
                agents_pose: agent positions in global coordinate
                alpha, beta, gamma, goal_w : Weights for Information Gain Function
        """
        belief = map_info.belief
        belief_frontier = map_info.belief_frontier
        map_mask = map_info.map_mask
        valid_leaves = []

        # TODO: Clearance 도입 고려
        # if add_clearance and dt is not None:
        # free_unknown = (~occ_mask_bel).astype(np.uint8)
        # edt_result = dt(free_unknown).astype(np.float32)
        # dt_m = edt_result * float(maps.res_m)
        # dt_norm = np.clip(dt_m, 0.0, 1.0)
        # else:
        #     # Clearance를 사용하지 않거나, dt 함수가 없음
        #     add_clearance = False # 강제로 비활성화
        #     dt_norm = None

        def _recursive_search(node: Node):
            # KD-Tree 계층 구조를 활용한 효율적인 재귀 Search
            r0, r1, c0, c1 = node.bounds

            has_frontier_in_bounds = np.any(
                belief_frontier[r0:r1, c0:c1] == map_mask["frontier"]
                )
            node.has_frontier = has_frontier_in_bounds

            if not has_frontier_in_bounds:
                # Frontier가 잡히지 않는 경우, 
                # 자식노드로 내려가봤자 Frontier 없으므로 탐색 중단
                return
            
            if node.is_leaf():
                # 해당 영역은 Valid 영역으로써, 이번 스텝에서 비용 업데이트 수행
                h, w = (r1-r0), (c1-c0)
                area = max(1, h*w)

                unk = np.sum(belief[r0:r1, c0:c1] == map_mask["unknown"]) / area
                occ = np.sum(belief[r0:r1, c0:c1] == map_mask["occupied"]) / area
                goal = np.sum(belief[r0:r1, c0:c1] == map_mask["goal"]) / area

                r = (r0 + r1) / 2
                c = (c0 + c1) / 2
                cx, cy = map_info.grid_to_world(r, c)

                d_list = [float(np.hypot(cx - ax, cy - ay)) for (ax, ay) in agents_pose]
                d_norm = np.mean(d_list)

                # Clearance (clr)
                # clr = 0.0
                # if add_clearance: # dt_norm은 바깥쪽 스코프에서 가져옴
                #     clr = float(np.mean(dt_norm[r0:r1, c0:c1]))

                J_now = alpha*unk - beta*occ - gamma * d_norm + goal_w*goal
                node.information_gain = J_now

                if J_now > self.valid_threshold:
                    # Leaf Node & Frontier 존재 + 임계값 이상
                    valid_leaves.append(node)
                return
            
            # 자식노드 재귀 탐색
            if node.left:
                _recursive_search(node.left)
            if node.right:
                _recursive_search(node.right)

        if self.root:
            # Root Node부터 재귀 탐색 시작
            _recursive_search(self.root)
        
        return valid_leaves