import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os

# ======================================================================================
# 1. Basic Utils & Layers
# ======================================================================================

def init_layer(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0)
    return layer

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(init_layer(nn.Linear(dims[i], dims[i+1])))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = init_layer(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = init_layer(nn.Conv2d(channels, channels, 3, padding=1))
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)

# ======================================================================================
# 2. Encoders & Logic Components
# ======================================================================================

class MapEncoder(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        # 입력이 이미 Pooling된 상태일 수 있으나, 추가적인 Feature Extraction을 위해 Conv 수행
        self.entry = nn.Sequential(
            init_layer(nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            init_layer(nn.Conv2d(32, feature_dim, 3, stride=2, padding=1)),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(
            ResidualBlock(feature_dim),
            ResidualBlock(feature_dim),
            ResidualBlock(feature_dim)
        )

    def forward(self, inputs):
        x = self.entry(inputs)
        x = self.blocks(x)
        return x 

    def extract_features_at(self, map_feat, coords):
        """
        map_feat: (Batch, C, H, W)
        coords: (Batch, N, 2) [row, col] - Must be scaled to feature map size
        """
        batch_size, n_points, _ = coords.shape
        _, c, h, w = map_feat.shape
        
        # 좌표 클리핑 (안전장치)
        rows = coords[..., 0].long().clamp(0, h-1)
        cols = coords[..., 1].long().clamp(0, w-1)
        
        # Batch Indexing
        batch_idx = torch.arange(batch_size, device=map_feat.device).view(-1, 1).expand(-1, n_points)
        
        # Gather: (Batch, N, C)
        features = map_feat[batch_idx, :, rows, cols] 
        return features


class BipartiteActor(nn.Module):
    def __init__(self, map_feat_dim, agent_extra_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Agent: Map Feature + [Row, Col] (Normalized)
        self.agent_embed = MLP(map_feat_dim + 2, [128], hidden_dim)
        # Frontier: Map Feature only
        self.frontier_embed = MLP(map_feat_dim, [128], hidden_dim) 
        
        self.W_Q = init_layer(nn.Linear(hidden_dim, hidden_dim))
        self.W_K = init_layer(nn.Linear(hidden_dim, hidden_dim))
        
        # 거리 가중치 (학습 가능)
        self.dist_weight = nn.Parameter(torch.tensor(5.0)) 

    def forward(self, agent_feat, frontier_feat, dist_matrix, mask):
        B, N_A, _ = agent_feat.shape
        _, N_F, _ = frontier_feat.shape
        head_dim = self.hidden_dim // self.num_heads

        # 1. Embeddings
        agent_emb = self.agent_embed(agent_feat)       # (B, N_A, H)
        frontier_emb = self.frontier_embed(frontier_feat) # (B, N_F, H)

        # 2. Attention Scores
        Q = self.W_Q(agent_emb).view(B, N_A, self.num_heads, head_dim).permute(0, 2, 1, 3)
        K = self.W_K(frontier_emb).view(B, N_F, self.num_heads, head_dim).permute(0, 2, 1, 3)

        # (B, Heads, N_A, N_F)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / (head_dim ** 0.5)
        
        # 3. Distance Bias Injection
        # dist_matrix: (B, N_A, N_F)
        # 거리가 멀수록(값이 클수록) 점수를 깎습니다.
        dist_bias = 0
        # dist_bias = - (dist_matrix.unsqueeze(1) * torch.abs(self.dist_weight))
        
        # 최종 Logits: Neural Score + Distance Bias
        final_logits = (scores + dist_bias).mean(dim=1) # Average over heads -> (B, N_A, N_F)

        # 4. Masking (Invalid Frontiers & Padding)
        if mask is not None:
            # mask가 1(True)인 곳을 -1e9로 채움
            final_logits = final_logits.masked_fill(mask, -1e9)

        return final_logits


class ContextualCritic(nn.Module):
    def __init__(self, map_feat_dim, agent_hidden_dim):
        super().__init__()
        self.map_conv = nn.Sequential(
            init_layer(nn.Conv2d(map_feat_dim, 32, 3, padding=1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten() # (B, 32)
        )
        self.agent_proj = MLP(agent_hidden_dim, [64], 32)
        self.value_head = MLP(32 + 32, [64, 32], 1)

    def forward(self, map_feat, agent_feat):
        # Global Context Extraction
        global_map = self.map_conv(map_feat) 
        # Agent Context Aggregation (Mean Pooling)
        global_agent = self.agent_proj(agent_feat).mean(dim=1) 
        
        combined = torch.cat([global_map, global_agent], dim=1)
        return self.value_head(combined)


# ======================================================================================
# 3. Main Network Wrapper (Handles Input Parsing & Batching)
# ======================================================================================

class MultiAgentActorCritic(nn.Module):
    def __init__(self, obs_shape, hidden_dim=128):
        super().__init__()
        self.input_ch = obs_shape[0] # 8 + Num_Agents
        
        # Map Encoder (8 ch input: Obstacle, Frontier, etc...)
        # Distance Map 채널은 별도로 처리하므로 Encoder에는 넣지 않거나 포함해도 무방
        # 여기서는 전체를 다 넣어서 Feature를 뽑습니다.
        self.map_encoder = MapEncoder(self.input_ch, feature_dim=64)
        
        self.actor = BipartiteActor(map_feat_dim=64, agent_extra_dim=2, hidden_dim=hidden_dim)
        self.critic = ContextualCritic(map_feat_dim=64, agent_hidden_dim=64+2)

    def forward(self, inputs, extras):
        """
        inputs: (B, 8 + N_A, H, W)
        extras: (B, 6 * N_A) -> Will be reshaped to (B, N_A, 6)
        """
        batch_size, _, H, W = inputs.shape
        
        # 1. Extras Parsing
        num_agents = (inputs.shape[1] - 8)
        extras = extras.view(batch_size, num_agents, 6)
        agent_coords = extras[:, :, :2] # (B, N_A, 2) [row, col]
        
        # Coordinate Scaling
        # inputs가 이미 downsampling된 맵이라면, agent_coords(원본 좌표)를 줄여야 함
        # 예: 원본 500, inputs 250 -> scale 0.5
        # 다만, ObservationManager에서 pooling_downsampling을 어떻게 썼느냐에 따라 다름.
        # 여기서는 inputs 크기(H)와 global_map_size 간의 비율을 추정하거나,
        # extras에 들어있는 map size 정보를 활용해야 함.
        # 안전하게: inputs 좌표계로 변환한다고 가정하고 비율 계산
        # (만약 inputs와 extras 좌표계가 같다면 scale=1.0)
        # 일단 1.0으로 가정하되, 필요시 수정 가능하도록 변수화
        scale_factor = 1.0 
        
        # 2. Frontier Extraction (Dynamic Batching)
        # Frontier Channel is index 1
        frontier_map = inputs[:, 1, :, :] # (B, H, W)
        
        # 배치 내에서 가장 많은 Frontier 개수 찾기 (Padding을 위해)
        # loop를 돌며 좌표 추출
        frontier_coords_list = []
        max_frontiers = 0
        
        for b in range(batch_size):
            f_coords = torch.nonzero(frontier_map[b] > 0) # (N_F, 2) [row, col]
            frontier_coords_list.append(f_coords)
            max_frontiers = max(max_frontiers, f_coords.size(0))
        
        if max_frontiers == 0:
            # Frontier가 아예 없는 경우 (Done 상태 등) -> Dummy 생성
            max_frontiers = 1
            for b in range(batch_size):
                frontier_coords_list[b] = torch.zeros((1, 2), device=inputs.device)

        # Padding & Stacking
        # padded_frontier_coords: (B, Max_F, 2)
        # frontier_mask: (B, N_A, Max_F) -> True if padding or unreachable
        padded_frontier_coords = torch.zeros((batch_size, max_frontiers, 2), device=inputs.device)
        padding_mask = torch.ones((batch_size, max_frontiers), device=inputs.device).bool() # 1=Padding
        
        for b in range(batch_size):
            nf = frontier_coords_list[b].size(0)
            padded_frontier_coords[b, :nf] = frontier_coords_list[b]
            padding_mask[b, :nf] = False # Valid parts are False
            
        # 3. Distance Sampling
        # inputs[:, 8:] contains distance maps for each agent
        # dist_maps: (B, N_A, H, W)
        dist_maps = inputs[:, 8:, :, :]
        
        # Gather distances at frontier locations
        # batch_idx: (B, 1, 1)
        # row/col: (B, 1, Max_F)
        # dist_matrix: (B, N_A, Max_F)
        
        rows = padded_frontier_coords[:, :, 0].long().clamp(0, H-1) # (B, Max_F)
        cols = padded_frontier_coords[:, :, 1].long().clamp(0, W-1) # (B, Max_F)
        
        # Advanced Indexing for Batch & Agent
        # We need to sample for each agent.
        # Let's expand dist_maps to gather efficiently or loop agents (N_A is small)
        
        dist_matrix_list = []
        for a in range(num_agents):
            # dist_map_a: (B, H, W)
            d_map = dist_maps[:, a, :, :]
            # Gather: (B, Max_F)
            # Use torch.gather or advanced indexing
            # Indexing: d_map[b, rows[b], cols[b]]
            
            # Creating Batch Indices
            b_idx = torch.arange(batch_size, device=inputs.device).unsqueeze(1).expand(-1, max_frontiers)
            
            d_vals = d_map[b_idx, rows, cols] # (B, Max_F)
            dist_matrix_list.append(d_vals)
            
        dist_matrix = torch.stack(dist_matrix_list, dim=1) # (B, N_A, Max_F)
        
        # 4. Feature Extraction & Embedding
        map_feat = self.map_encoder(inputs)
        
        agent_map_feat = self.map_encoder.extract_features_at(map_feat, agent_coords * scale_factor)
        frontier_map_feat = self.map_encoder.extract_features_at(map_feat, padded_frontier_coords)
        
        # Agent: Normalize coords for appending
        norm_agent_coords = agent_coords / max(H, W)
        agent_full_feat = torch.cat([agent_map_feat, norm_agent_coords], dim=-1) # (B, N_A, 64+2)
        
        # 5. Mask Construction
        # Padding Mask 확장: (B, 1, Max_F) -> (B, N_A, Max_F)
        # Distance Mask: 거리 4.0 이상이면 Unreachable (user config)
        
        final_mask = padding_mask.unsqueeze(1).expand(-1, num_agents, -1) | (dist_matrix >= 4.0)
        
        # 6. Forward Passes
        logits = self.actor(agent_full_feat, frontier_map_feat, dist_matrix, final_mask)
        value = self.critic(map_feat, agent_full_feat).squeeze(0)
        
        return value, logits


# ======================================================================================
# 4. RL Policy (Integration)
# ======================================================================================

class RL_Policy(nn.Module):
    def __init__(self, obs_shape, action_space, model_type='gnn',
                 base_kwargs=None, lr=None, eps=None):
        super(RL_Policy, self).__init__()
        
        self.network = MultiAgentActorCritic(obs_shape, hidden_dim=128)
        self.model_type = model_type

        # Optimizer Setup
        # Critic & MapEncoder -> Critic Optimizer
        # Actor -> Actor Optimizer
        actor_params = list(self.network.actor.parameters())
        critic_params = list(self.network.critic.parameters()) + list(self.network.map_encoder.parameters())

        self.actor_optimizer = optim.Adam(actor_params, lr=lr[0], eps=eps)
        self.critic_optimizer = optim.Adam(critic_params, lr=lr[1], eps=eps)

    def forward(self, inputs, rnn_hxs, masks, extras):
        return self.network(inputs, extras)

    def act(self, inputs, rnn_hxs, masks, extras, deterministic=False):
        # inputs: (B, 8+N_A, H, W)
        # extras: (B, 6*N_A)
        
        value, logits = self.network(inputs, extras)
        
        # logits: (B, N_A, Max_F) -> 우리는 각 Agent가 하나의 Frontier를 선택
        # 
        
        # Multi-Categorical Distribution
        # 각 Agent별로 독립적인 Categorical 분포 생성
        dist = Categorical(logits=logits)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(dim=-1)
        
        # rnn_hxs는 사용하지 않지만 인터페이스 유지를 위해 반환
        return value, action, action_log_probs, rnn_hxs, dist.probs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _ = self.network(inputs, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):
        value, logits = self.network(inputs, extras)
        dist = Categorical(logits=logits)

        action_log_probs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist.probs

    # ---------------- Save / Load ----------------
    def load(self, path, device):
        # 파라미터 로드 전 Optimizer 초기화 (LR 초기화 효과)
        actor_params = list(self.network.actor.parameters())
        critic_params = list(self.network.critic.parameters()) + list(self.network.map_encoder.parameters())
        
        self.actor_optimizer = optim.Adam(actor_params, lr=1e-3)
        self.critic_optimizer = optim.Adam(critic_params, lr=1e-3)

        state_dict = torch.load(path, map_location=device)
        self.network.load_state_dict(state_dict['network'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        del state_dict

    def load_critic(self, path, device):
        state_dict = torch.load(path, map_location=device)['network']
        critic_state = {k.replace('critic.', ''): v for k, v in state_dict.items() if 'critic.' in k}
        encoder_state = {k.replace('map_encoder.', ''): v for k, v in state_dict.items() if 'map_encoder.' in k}
        
        self.network.critic.load_state_dict(critic_state)
        self.network.map_encoder.load_state_dict(encoder_state)
        del state_dict

    def save(self, path):
        state = {
            'network': self.network.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)