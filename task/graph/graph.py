import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import deque

class ConnectivityGraph:
    """
        여러 에이전트의 위치를 기반으로 최소 스패닝 트리(MST)를 계산하고,
        각 에이전트의 연결성 유지 대상('부모' 노드)을 관리하는 클래스.
    """
    def __init__(self, num_agents: int):
        """
            Args:
                num_agents: 시뮬레이션에 참여하는 에이전트의 수
        """
        if num_agents <= 1:
            raise ValueError("에이전트의 수는 2 이상이어야 합니다.")
        self.num_agents = num_agents
        # 각 에이전트의 부모 ID를 저장할 배열. -1은 부모가 없음을 의미(루트 노드).
        self.parents = np.full(num_agents, -1, dtype=int)

    def update_and_compute_mst(self, agent_positions: np.ndarray, root_agent_id: int = 0):
        """
            에이전트 위치를 업데이트하고, MST를 다시 계산하여 부모-자식 관계를 설정
            Inputs:
                agent_positions: 에이전트들의 현재 위치 배열. shape: (num_agents, 2)
                root_agent_id: 트리의 루트로 지정할 에이전트의 ID
        """
        if agent_positions.shape[0] != self.num_agents:
            raise ValueError(f"입력된 위치의 수({agent_positions.shape[0]})가 "
                             f"초기화된 에이전트 수({self.num_agents})와 다릅니다.")

        # 1. 완전 그래프 구성: 모든 에이전트 쌍 사이의 거리를 계산하여 인접 행렬 생성
        adjacency_matrix = cdist(agent_positions, agent_positions)
        # 2. 최소 스패닝 트리(MST) 계산
        mst = minimum_spanning_tree(adjacency_matrix)
        # 3. 부모-자식 관계 설정 (너비 우선 탐색 - BFS)
        self.parents.fill(-1)
        # BFS를 위한 큐(queue)와 방문 기록(visited)
        queue = deque([root_agent_id])
        visited = {root_agent_id}
        
        # 희소 행렬을 양방향 그래프처럼 탐색하기 위해 coo_matrix로 변환
        mst_coo = mst.tocoo()
        # 각 노드에 연결된 이웃들을 쉽게 찾기 위한 인접 리스트 생성
        adj_list = [[] for _ in range(self.num_agents)]
        for r, c, _ in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            adj_list[r].append(c)
            adj_list[c].append(r)

        while queue:
            current_agent = queue.popleft()
            
            for neighbor in adj_list[current_agent]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    # current_agent가 neighbor의 부모가 됨
                    self.parents[neighbor] = current_agent
                    queue.append(neighbor)
        
        # 모든 에이전트가 연결되었는지 확인 (디버깅용)
        if len(visited) != self.num_agents:
            print("[Warning] 모든 에이전트가 MST에 포함되지 않았습니다. 그래프가 분리되었을 수 있습니다.")


    def get_parent(self, agent_id: int) -> int | np.ndarray:
        """
        특정 에이전트의 부모 에이전트 ID를 반환합니다.

        :param agent_id: 부모를 찾을 에이전트의 ID
        :return: 부모 에이전트의 ID. 루트 노드일 경우 -1을 반환.
        """
        if not (0 <= agent_id < self.num_agents):
            raise IndexError("유효하지 않은 에이전트 ID입니다.")
        return self.parents[agent_id]
    
    def get_child(self, agent_id: int) -> np.ndarray | None:
        """
        특정 에이전트를 부모로 갖는 자식 에이전트 ID를 반환합니다.
        """
        if not (0 <= agent_id < self.num_agents):
            raise IndexError("유효하지 않은 에이전트 ID입니다.")
        ids = np.where(self.parents == agent_id)[0]

        if ids.size > 0:
            return ids
        else:
            # Leaf Node에 대해서는 -1 반환
            return None