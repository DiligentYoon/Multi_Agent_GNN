from copy import deepcopy
import torch
import os
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical





class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def MLP(channels, do_bn=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PositionalEncoder(nn.Module):
    """
    Sinusoidal positional encoding for spatial coordinates.
    (From NeRF: https://github.com/yenchenlin/nerf-pytorch)
    """
    def __init__(self, num_freqs: int, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = True
        
        if self.log_sampling:
            self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(1., 2.**(num_freqs - 1), steps=num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape [..., D]
        out = []
        if self.include_input:
            out.append(x)
        
        for freq in self.freq_bands.to(x.device):
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
            
        return torch.cat(out, dim=-1)


class Encoder(nn.Module):
    """ Joint encoding of semantic tag and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        
        num_pos_freqs = 10 # Number of frequency bands for positional encoding
        # Calculate the output dimension of the positional encoder
        # D_in=2 (x,y), include_input=True, so (2) + (2 * 2 * num_pos_freqs)
        pos_enc_dim = 2 + (2 * 2 * num_pos_freqs)
        # The final input to the MLP is the positional encoding + 2D one-hot type
        encoder_input_dim = pos_enc_dim + 2

        self.pos_encoder = PositionalEncoder(num_pos_freqs, include_input=True)
        
        self.frontier_encoder = MLP([encoder_input_dim] + layers + [feature_dim])
        self.agent_encoder = MLP([encoder_input_dim] + layers + [feature_dim])
        self.dist_encoder = MLP([1, feature_dim, feature_dim])

        nn.init.constant_(self.frontier_encoder[-1].bias, 0.0)
        nn.init.constant_(self.agent_encoder[-1].bias, 0.0)
        nn.init.constant_(self.dist_encoder[-1].bias, 0.0)

    def _create_feature_batch(self, points, type_vec, sz_r, sz_c):
        if points.size(0) == 0:
            return None
        
        # Normalize coordinates to [0, 1]
        norm_coords = torch.stack([
            points.float()[:, 0] / sz_r,
            points.float()[:, 1] / sz_c
        ], dim=-1)
        
        # Create positional encoding
        pos_enc = self.pos_encoder(norm_coords)
        
        # Create type encoding
        type_enc = type_vec.expand(points.size(0), 2)
        
        # Concatenate and format for Conv1d
        features = torch.cat([pos_enc, type_enc], dim=-1).unsqueeze(0) # Shape: [1, num_points, final_dim]
        return features.transpose(1, 2) # Shape: [1, final_dim, num_points]

    def forward(self, inputs, dist, pos_history, goal_history, extras):
        inputs = inputs[:, 1, :, :]
        sz_r = inputs.size(1)
        sz_c = inputs.size(2)
        
        frontier_idxs = []
        frontier_batches = []
        agent_batches = []
        dist_batches = []
        phistory_idxs = []
        phistory_batches = []
        ghistory_idxs = []
        ghistory_batches = []

        device = inputs.device
        type_frontier = torch.tensor([1., 0.], device=device)
        type_agent = torch.tensor([0., 1.], device=device)

        for b in range(inputs.size(0)):
            frontier = torch.nonzero(inputs[b, :, :])

            if frontier.size(0) > 0:
                dist_structured = dist[b, :, :, :][(inputs[b, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(dist.size(1), -1)

                perm = torch.randperm(frontier.size(0), device=device)
                frontier = frontier[perm]
                dist_structured = dist_structured[:, perm]
                
                dist_feat = torch.log1p(dist_structured.reshape(1, 1, -1))
            else:
                dist_feat = torch.empty(1, 1, 0, device=device)

            frontier_idxs.append(frontier)
            dist_batches.append(dist_feat)
            
            # Create feature batches using the helper function
            frontier_batches.append(self._create_feature_batch(frontier, type_frontier, sz_r, sz_c))

            if pos_history is not None:
                phistory_pos = torch.nonzero(pos_history[b, :, :])
                phistory_idxs.append(phistory_pos)
                phistory_batches.append(self._create_feature_batch(phistory_pos, type_agent, sz_r, sz_c))
            
            if goal_history is not None:
                ghistory_pos = torch.nonzero(goal_history[b, :, :])
                ghistory_idxs.append(ghistory_pos)
                ghistory_batches.append(self._create_feature_batch(ghistory_pos, type_frontier, sz_r, sz_c))

            agent_pos = extras[b, :, :2].long()
            agent_batches.append(self._create_feature_batch(agent_pos, type_agent, sz_r, sz_c))
        
        # Helper to handle None in list comprehension
        def encode_batch(encoder, batch_list):
            return [(encoder(batch) if batch is not None else None) for batch in batch_list]

        return (
            frontier_idxs,
            phistory_idxs if pos_history is not None else ([None] * len(frontier_idxs)),
            ghistory_idxs if goal_history is not None else ([None] * len(frontier_idxs)),
            [self.dist_encoder(batch) for batch in dist_batches],
            encode_batch(self.frontier_encoder, frontier_batches),
            encode_batch(self.agent_encoder, agent_batches),
            encode_batch(self.agent_encoder, phistory_batches) if pos_history is not None else ([None] * len(frontier_idxs)),
            encode_batch(self.frontier_encoder, ghistory_batches) if goal_history is not None else ([None] * len(frontier_idxs))
        )
    

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def attention(self, query, key, value, mask):
        dim = query.shape[1]
        scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
        if mask is not None:
            scores = scores + (scores.min().detach()) * (~mask).float().unsqueeze(0).unsqueeze(0).repeat(1, self.num_heads, 1, 1)
        prob = torch.nn.functional.softmax(scores, dim=-1)
        return torch.einsum('bhnm,bdhm->bdhn', prob, value), scores

    def forward(self, query, key, value, dist, mask):
        query, key, value = [l(x).view(1, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, scores = self.attention(query, key, value, mask)
        return self.merge(x.contiguous().view(1, self.dim*self.num_heads, -1)), scores.mean(1)


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, type: str):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, dist, mask):
        message, weights = self.attn(x, source, source, dist, mask)
        return self.mlp(torch.cat([x, message], dim=1)), weights


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, use_history: bool, ablation: int):
        super().__init__()
        self.attn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, type) for type in layer_names])
        self.norm0 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in layer_names])
        self.norm1 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in layer_names])

        if use_history:
            self.phattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
            self.ghattn = nn.ModuleList([AttentionalPropagation(feature_dim, 4, 'self') for type in layer_names])
            self.norm2 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in layer_names])
            self.norm3 = nn.ModuleList([nn.LayerNorm(feature_dim) for _ in layer_names])
        else:
            self.phattn = [None for type in layer_names]
            self.ghattn = [None for type in layer_names]
        # self.attn = MLP([feature_dim, 1])
        self.use_history = use_history
        self.score_layer = MLP([2*feature_dim, feature_dim, 1])
        self.names = layer_names
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.ablation = ablation

    def forward(self, desc0, desc1, desc2, desc3, lmb, fidx, phidx, ghidx, dist, unreachable):
        # desc0: frontier
        # desc1: agent
        # fidx: n_frontier x 2
        # lmb: n_agent x 4

        if self.ablation != 2:

            dist0 = dist.view(-1, desc1.size(-1), desc0.size(-1)).transpose(1, 2).reshape(1, -1, desc1.size(-1) * desc0.size(-1))
            dist1 = dist

            for idx, attn, phattn, ghattn, name in zip(range(len(self.names)), self.attn, self.phattn, self.ghattn, self.names):

                if name == 'cross':
                    src0, src1 = desc1, desc0
                else:
                    src0, src1 = desc0, desc1

                delta0, score0 = attn(desc0, src0, dist0, None)
                delta1, score1 = attn(desc1, src1, dist1, None)

                if self.use_history:
                    if name == 'cross':
                        if desc2 is not None:
                            delta21, _ = phattn(desc2, desc1, None, None)
                            delta12, _ = phattn(desc1, desc2, None, None)
                            desc2 = desc2 + delta21
                            desc2 = self.norm2[idx](desc2.transpose(1, 2)).transpose(1, 2)
                        else:
                            delta12 = 0
                        if desc3 is not None:
                            delta03, _ = ghattn(desc0, desc3, None, None)
                            delta30, _ = ghattn(desc3, desc0, None, None)
                            desc3 = desc3 + delta30
                            desc3 = self.norm3[idx](desc3.transpose(1, 2)).transpose(1, 2)
                        else:
                            delta03 = 0
                        desc0, desc1 = (desc0 + delta0 + delta03), (desc1 + delta1 + delta12)
                    else:  # if name == 'self':
                        if desc2 is not None:
                            delta2, _ = phattn(desc2, desc2, None, None)
                            desc2 = desc2 + delta2
                            desc2 = self.norm2[idx](desc2.transpose(1, 2)).transpose(1, 2)
                        if desc3 is not None:
                            delta3, _ = ghattn(desc3, desc3, None, None)
                            desc3 = desc3 + delta3
                            desc3 = self.norm3[idx](desc3.transpose(1, 2)).transpose(1, 2)
                        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
                else:
                    desc0, desc1 = (desc0 + delta0), (desc1 + delta1)

                desc0 = self.norm0[idx](desc0.transpose(1, 2)).transpose(1, 2)
                desc1 = self.norm1[idx](desc1.transpose(1, 2)).transpose(1, 2)

        # weights1: n_agent x n_frontier
        fidx = torch.repeat_interleave(fidx.view(1, fidx.size(0), 2), repeats=lmb.size(0), dim=0)
        lmb = torch.repeat_interleave(lmb.view(lmb.size(0), 1, 4), repeats=fidx.size(1), dim=1)
        # fidx: n_agent x n_frontier x 2
        # lmb: n_agent x n_frontier x 4
        # unreachable: n_agent x n_frontier
        invalid = ((fidx < lmb[:, :, [0,2]]) | (fidx >= lmb[:, :, [1,3]])).any(2)
        # assert (~invalid).any(1).all()
        if self.ablation == 1:
            scores = self.score_layer(torch.cat((
                torch.repeat_interleave(desc1, repeats=unreachable.size(1), dim=-1),
                desc0.repeat(1, 1, unreachable.size(0))
            ), dim=1)).view(1, *unreachable.shape)
        elif self.ablation == 2:
            scores = 2 / (dist.view(1, *unreachable.shape) + 1e-3)
        else:
            scores = score1

        if not torch.isfinite(scores).all():
            print("[NaN] scores BEFORE norm has NaN/inf")
            print(scores)
            raise RuntimeError

        # scores = scores.log_softmax(dim=-2).view(unreachable.shape)
        # scores = log_optimal_transport(scores.log_softmax(dim=-2), self.bin_score, iters=5)[:, :-1, :-1].view(unreachable.shape)
        scores = scores.mean(dim=1).view(-1)
        # Apply unreachable mask
        unreachable_frontiers = unreachable.any(dim=0)
        scores.masked_fill_(unreachable_frontiers, -1e9)

        return scores


class Actor(nn.Module):
    def __init__(self, desc_dim, gnn_layers, use_history, ablation):
        super().__init__()
        self.kenc = Encoder(desc_dim, [32, 64, 128, 256])
        self.gnn = AttentionalGNN(desc_dim, gnn_layers, use_history, ablation)
        self.ablation = ablation

    def forward(self, inputs, dist, pos_history, goal_history, extras):
        # MLP encoder.
        extras = extras.view(inputs.size(0), -1, 6)
        unreachable = [
            dist[b, :, :, :][(inputs[b, 1, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(dist.size(1), -1) >= 4
            for b in range(inputs.size(0))
        ]

        if self.ablation == 2:
            idxs, phidxs, ghidxs, _, desc0s, desc1s, desc2s, desc3s = self.kenc(inputs, dist, pos_history, goal_history, extras[:, :, :2])
            dist = [
                dist[b, :, :, :][(inputs[b, 1, :, :] > 0).unsqueeze(0).repeat(dist.size(1), 1, 1)].view(dist.size(1), -1)
                for b in range(inputs.size(0))
            ]
        else:
            idxs, phidxs, ghidxs, dist, desc0s, desc1s, desc2s, desc3s = self.kenc(inputs, dist, pos_history, goal_history, extras[:, :, :2])

        if torch.isfinite(torch.cat(dist, dim=-1)).all() == False:
            print("[NaN] dist has NaN/inf")
            raise RuntimeError

        # Multi-layer Transformer network.
        return [self.gnn(desc0s[b], desc1s[b], desc2s[b], desc3s[b], extras[b, :, 2:], idxs[b], phidxs[b], ghidxs[b], dist[b], unreachable[b]) for b in range(inputs.size(0))]


class GNN(nn.Module):
    def __init__(self, input_shape, gnn_layers, use_history, ablation):
        super().__init__()
        self.output_size = 0
        self.is_recurrent = False
        self.rec_state_size = 1
        self.downscaling = 1
        # desc_dim = 128
        desc_dim = 32
        self.actor = Actor(desc_dim, gnn_layers, use_history, ablation)

        self.critic_encoder = nn.Sequential(
            nn.Conv2d(6, 64, 6, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 6, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 6, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Flatten()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 6, input_shape[1], input_shape[2])
            flatten_size = self.critic_encoder(dummy_input).shape[1]

        self.critic_layer = nn.Sequential(
            nn.Linear(flatten_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.critic = nn.Sequential(
            self.critic_encoder,
            self.critic_layer
        )

        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        value = self.critic(inputs[:, :6, :, :]).squeeze(-1)
        actor_features = self.actor(inputs[:, :6, :, :], inputs[:, 8:, :, :], inputs[:, 6, :, :], inputs[:, 7, :, :], extras)
        return value, actor_features, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_ActorCritic(nn.Module):

    def __init__(self, obs_shape, action_space, model_type='gconv',
                 base_kwargs=None, lr=None, eps=None):

        super(RL_ActorCritic, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        self.network = GNN(obs_shape, 
                           base_kwargs.get('num_gnn_layer') * ['self', 'cross'], 
                           base_kwargs.get('use_history'), base_kwargs.get('ablation'))
    
        assert action_space.__class__.__name__ == "Box"
        self.num_action = action_space.shape[0]

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr[0], eps=eps)

        self.model_type = model_type


    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    @property
    def downscaling(self):
        return self.network.downscaling

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)


    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        
        # Pad the list of logit tensors
        if not actor_features or all(f.numel() == 0 for f in actor_features):
            batch_size = inputs.size(0)
            action = torch.full((batch_size, self.num_action), -1, dtype=torch.long, device=inputs.device)
            action_log_probs = torch.zeros(batch_size, device=inputs.device)
            return value, action, action_log_probs, rnn_hxs, actor_features

        max_len = max((f.shape[0] for f in actor_features if f.numel() > 0), default=0)
        if max_len == 0:
            batch_size = inputs.size(0)
            action = torch.full((batch_size, self.num_action), -1, dtype=torch.long, device=inputs.device)
            action_log_probs = torch.zeros(batch_size, device=inputs.device)
            return value, action, action_log_probs, rnn_hxs, actor_features

        padded_logits = []
        for logits in actor_features:
            if logits.numel() > 0:
                pad_size = max_len - logits.shape[0]
                padded = F.pad(logits, (0, pad_size), 'constant', -1e9)
                padded_logits.append(padded)
            else:
                padded_logits.append(torch.full((max_len,), -1e9, device=inputs.device, dtype=torch.float))
        
        logits_tensor = torch.stack(padded_logits, dim=0)
        
        dist = Categorical(logits=logits_tensor)

        if deterministic:
            _, action = torch.topk(logits_tensor, k=self.num_action, dim=-1)
            valid_count = logits_tensor.size(-1)
            k_to_select = min(self.num_action, valid_count)
            
            _, selected_actions = torch.topk(logits_tensor, k=k_to_select, dim=-1)
            
            #  목표 개수보다 적게 뽑힌경우, 부족한 만큼 가장 높은 타겟으로 채움
            if k_to_select < self.num_action:
                num_missing = self.num_action - k_to_select
                
                best_action = selected_actions[:, 0:1]
                
                padding = best_action.expand(-1, num_missing)
                
                action = torch.cat([selected_actions, padding], dim=-1)
            else:
                action = selected_actions
        else:
            # sample_shape prepends to batch_shape, so result is (num_action, batch_size)
            action_transposed = dist.sample(sample_shape=torch.Size([self.num_action]))
            action = action_transposed.transpose(0, 1)

        # Calculate log_probs for the chosen action
        action_transposed_for_log_prob = action.transpose(0, 1)
        log_probs_transposed = dist.log_prob(action_transposed_for_log_prob)
        action_log_probs = log_probs_transposed.sum(dim=0)

        return value, action, action_log_probs, rnn_hxs, dist.probs


    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, actor_features, _ = self(inputs, rnn_hxs, masks, extras)
        return value, actor_features

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        
        # Pad the list of logit tensors
        if not actor_features or all(f.numel() == 0 for f in actor_features):
            batch_size = inputs.size(0)
            action_log_probs = torch.zeros(batch_size, device=inputs.device)
            dist_entropy = torch.zeros(1, device=inputs.device).mean()
            return value, action_log_probs, dist_entropy, rnn_hxs, actor_features

        max_len = max((f.shape[0] for f in actor_features if f.numel() > 0), default=0)
        if max_len == 0:
            batch_size = inputs.size(0)
            action_log_probs = torch.zeros(batch_size, device=inputs.device)
            dist_entropy = torch.zeros(1, device=inputs.device).mean()
            return value, action_log_probs, dist_entropy, rnn_hxs, actor_features

        padded_logits = []
        for logits in actor_features:
            if logits.numel() > 0:
                pad_size = max_len - logits.shape[0]
                padded = F.pad(logits, (0, pad_size), 'constant', -1e9)
                padded_logits.append(padded)
            else:
                padded_logits.append(torch.full((max_len,), -1e9, device=inputs.device, dtype=torch.float))

        logits_tensor = torch.stack(padded_logits, dim=0)
        
        dist = Categorical(logits=logits_tensor)

        # Mask out invalid actions (-1) before calculating log_prob
        action_mask = action >= 0
        action_transposed = action.transpose(0, 1)

        # Use clamp to avoid error on -1 index
        log_probs_transposed = dist.log_prob(action_transposed.clamp(min=0))

        # Zero out log_probs for padded actions and sum
        log_probs_transposed = log_probs_transposed * action_mask.transpose(0, 1).float()
        action_log_probs = log_probs_transposed.sum(dim=0)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, dist.probs

    def load_critic(self, path, device):
        state_dict = torch.load(path, map_location=device)['network']
        self.network.critic.load_state_dict({k.replace('critic.', ''):v for k,v in state_dict.items() if 'critic' in k})
        # self.network.actor.load_state_dict({k.replace('actor.', ''):v for k,v in state_dict.items() if 'actor' in k})
        del state_dict


    def get_policy_data(self):
        """
        Returns the state of the agent for checkpointing.
        """
        return self.network.actor.state_dict()