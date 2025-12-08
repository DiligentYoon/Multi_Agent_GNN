# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import logging

def get_rotation_mat(theta):
    theta = torch.tensor(theta * 3.14159265359 / 180.)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0.],
                         [torch.sin(theta), torch.cos(theta), 0.]]).cuda()


def rotate_tensor(x, theta):
    rot_mat = torch.repeat_interleave(get_rotation_mat(theta).unsqueeze(0), repeats=x.size(0), dim=0)
    grid = F.affine_grid(rot_mat, x.shape)
    x = F.grid_sample(x, grid)
    return x


def rotate_scalar(x, theta, map_size):
    origin = (map_size - 1) / 2
    theta = torch.tensor(theta * 3.14159265359 / 180.)
    x, y = (x // map_size - origin).float(), (x % map_size - origin).float()
    x, y = torch.cos(theta) * x - torch.sin(theta) * y + origin, torch.sin(theta) * x + torch.cos(theta) * y + origin
    x, y = torch.clamp(x.long(), 0, map_size - 1), torch.clamp(y.long(), 0, map_size - 1)
    return x * map_size + y



class RolloutBuffer(object):
    def __init__(self, 
                 rollouts, 
                 num_envs, 
                 eval_freq, 
                 num_repeats, 
                 num_agents, 
                 obs_shape, 
                 action_space, 
                 rec_state_size):
        """
        PPO와 같은 On-Policy 알고리즘의 학습 데이터를 모으는 RolloutBuffer 클래스

            Args:
                rollouts: 한번에 모을 환경 스텝 수
                num_envs: 병렬 환경 수 (현 프로젝트에서는 1)
                eval_freq: 학습 도중, 평가를 수행하는 주기
                num_repeats: GAE를 위한 rollout 덩어리 개수
                obs_shape: shape of observation space
                action_space: action space
                rec_state_size: reccurent neural network (RNN)을 위한 추가 state size (현 프로젝트에서는 1)
            
            Components:
                Buffer는 아래와 같이 구성되며 Rollout 차원에 +1은 GAE 계산을 위해서 추가.
                1. obs: agent의 observation을 저장하는 텐서 [R+1, m, n_o] <-- m : 한 Rollout step에 쌓이는 데이터 개수
                2. rec_states: RNN을 위한 staet를 저장하는 텐서 [R+1, m, 1] <-- 사용 X
                3. rewards: 보상을 저장하는 텐서 [R+1, m, ]
                4. value_preds: next_q_value를 저장하는 텐서 [R+1, m, ]
                5. action_log_probs: 추출한 action에 대한 log_probability를 저장하는 텐서 [R, m] (stochastic policy 업데이트에 사용)
                6. actions: 수행한 action을 저장하는 텐서 [R, m, n_a]
                7. mask: 현 스텝에서의 에피소드 종료 여부 [R+1, m, ] (next_q_value 계산할 때 사용) 
                8. open: Rollout동안 에피소드 종료 여부 [m, ] (통계 분석)
        """
        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            self.map_size = int(action_space.n ** 0.5)
            action_type = torch.long
        else:
            self.map_size = 0
            self.n_actions = action_space.shape[0]
            action_type = torch.float32 if action_space.dtype == 'float32' else torch.long

        self.obs = torch.zeros(rollouts + 1, num_envs * num_agents * num_repeats, *obs_shape)
        self.rec_states = torch.zeros(rollouts + 1, num_envs * num_agents * num_repeats, rec_state_size)
        self.rewards = torch.zeros(rollouts, num_envs * num_agents * num_repeats)
        self.value_preds = torch.zeros(rollouts + 1, num_envs * num_agents * num_repeats)
        self.returns = torch.zeros(rollouts + 1, num_envs * num_agents * num_repeats)
        self.action_log_probs = torch.zeros(rollouts, num_envs * num_agents * num_repeats)
        self.actions = torch.zeros((rollouts, num_envs * num_agents * num_repeats, self.n_actions), dtype=action_type)
        self.masks = torch.ones(rollouts + 1, num_envs * num_agents * num_repeats)
        self.open = torch.ones(num_envs * num_agents * num_repeats).bool()

        self.num_rollout_blocks = num_repeats
        self.mini_step_size = num_envs * num_agents
        self.rollouts = rollouts
        self.step = 0
        self.mini_step = 0
        self.has_extras = False
        self.extras_size = None
        self.first_use_to_eval = (eval_freq == num_repeats and num_repeats > 1)


    def to(self, device):
        """
        모든 Buffer Components들의 device를 변경

            Inputs:
                device : ["cpu", "cuda"]
        """
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.open = self.open.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self


    def insert(self, 
               obs, 
               rec_states, 
               actions, 
               action_log_probs, 
               value_preds,
               rewards, 
               masks):
        """
        환경으로부터 얻은 데이터 & 예측치를 버퍼에 순차적으로 저장
            Inputs:
                obs: observation at t+1
                rec_states: recurrent state at t+1
                actions: action at t
                action_log_probs: log probability of action at t
                value_preds: predicted next q value at t
                rewards: reward at t+1
                masks: done mask at t+1
        """
        l, h = self.mini_step * self.mini_step_size, (self.mini_step + 1) * self.mini_step_size
        if self.step == 0:
            ll, lh = l-self.mini_step_size, h-self.mini_step_size
            if lh == 0:
                lh = self.mini_step_size * self.num_rollout_blocks
            self.obs[0][l:h].copy_(self.obs[-1][ll:lh])
            self.rec_states[0][l:h].copy_(self.rec_states[-1][ll:lh])
        self.obs[self.step + 1][l:h].copy_(obs)
        self.rec_states[self.step + 1][l:h].copy_(rec_states)
        self.actions[self.step][l:h].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step][l:h].copy_(action_log_probs)
        self.value_preds[self.step][l:h].copy_(value_preds)
        self.rewards[self.step][l:h].copy_(rewards)
        self.masks[self.step + 1][l:h].copy_(masks)
        self.open[l:h] = self.open[l:h] & masks.bool()

        self.step = (self.step + 1) % self.rollouts
        if self.step == 0:
            self.mini_step = (self.mini_step + 1) % self.num_rollout_blocks


    def after_update(self):
        """
        Update 이후, Buffer 초기화
        """
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.open[:] = True
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])


    def compute_returns(self, 
                        next_value, 
                        use_gae, 
                        gamma, 
                        lam):
        """
        모든 스텝에 대하여 Generalized Advantage Expectation (GAE)를 계산 후, 버퍼에 저장

            Inputs:
                next_value: predicted next q value at last step
                use_gae: whether use gae or not
                gamma: discount factor
                lam: gae lambda factor
        """
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * lam * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * gamma \
                                     * self.masks[step + 1] + self.rewards[step]


    def sample_mini_batch(self, advantages, num_mini_batch, max_batch_size, rotation_augmentation, ds=1, verbose=True):
        """
        파라미터 업데이트를 위해 Buffer에서 데이터를 샘플링

            Inputs:
                advantages: GAE로부터 얻은 값과 기존 Value값을 통해 계산한 Advantages
                num_mini_batch: 총 Batch 개수
                max_batch_size: -1, 전체 Batch 모두 사용
                rotation_augmentation: 맵 데이터에 대한 Rotation을 걸어, Data Augmentation을 수행할지에 대한 여부
                ds : action에 대한 rotation aumentation을 수행할 때, State DownScaling (MaxPool)에 맞춰 액션을 보정하기 위한 상수
                     -> However, Centralized 방식에서는 Frontier Node선택이 액션이므로 다운 스케일링 보정 필요 X (값 = 1)
                verbose : logging 여부
        """
        rollouts = self.rollouts
        num_data_per_rollout = self.mini_step_size * self.num_rollout_blocks
        batch_size = num_data_per_rollout * rollouts
        batch_begin = self.mini_step_size * rollouts if self.first_use_to_eval else 0
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_data_per_rollout, rollouts, num_data_per_rollout * rollouts,
                      num_mini_batch))
        idx = [i for i in range(batch_begin, batch_size) if self.masks[i // num_data_per_rollout, i % num_data_per_rollout]]
        # idx = range(batch_size)
        if verbose:
            logging.info(f"actual-batch-size: {len(idx)}/{batch_size - batch_begin}")
            logging.info(f"open-ratio: {self.open.sum()}/{self.open.size(0)}")
        # 하나의 큰 배치에 총 몇개의 미니배치가 들어갈지 계산
        # NOTE: num_mini_batch = 4, idx = 대부분 rollout * num_envs * num_agents * num_repeats  ===> mini_batch_size = 3 ===> 즉, 하나의 큰 배치에 미니배치 3개 들어감 
        mini_batch_size = len(idx) // num_mini_batch
        if max_batch_size > 0:
            mini_batch_size = min(max_batch_size, mini_batch_size)
        if mini_batch_size > 0:
            sampler = BatchSampler(SubsetRandomSampler(idx),
                                    mini_batch_size, drop_last=False)

            for idx, indices in enumerate(sampler):
                if idx >= num_mini_batch:
                    break
                # [Rollout * num_envs * num_agents * num_repeats] 를 배치차원으로 하여 랜덤 샘플링
                raw_data = {
                    'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                    'rec_states': self.rec_states[:-1].view(-1, self.rec_states.size(-1))[indices],
                    'actions': self.actions.view(-1, self.n_actions)[indices],
                    'value_preds': self.value_preds[:-1].view(-1)[indices],
                    'returns': self.returns[:-1].view(-1)[indices],
                    'masks': self.masks[:-1].view(-1)[indices],
                    'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                    'adv_targ': advantages.view(-1)[indices],
                    'extras': self.extras[:-1].view(-1, self.extras_size)[indices] if self.has_extras else None,
                    'augmentation': False
                }
                yield raw_data
                if rotation_augmentation > 1:
                    raw_data['augmentation'] = True
                    raw_obs = raw_data['obs']
                    raw_actions = raw_data['actions']
                    for i in range(1, rotation_augmentation):
                        raw_data['obs'] = rotate_tensor(raw_obs, i * 360. / rotation_augmentation)
                        raw_data['actions'] = rotate_scalar(raw_actions, i * 360. / rotation_augmentation, self.map_size // ds)
                        x = raw_data['actions'] // (self.map_size // ds)
                        y = raw_data['actions'] % (self.map_size // ds)
                        raw_data['obs'][:, 1, x * ds, y * ds] = 1.
                        yield raw_data



class CoMappingRolloutBuffer(RolloutBuffer):
    def __init__(self, 
                 rollouts, 
                 num_envs, 
                 eval_freq, 
                 num_repeats, 
                 num_agents, 
                 obs_shape, 
                 action_space,
                 rec_state_size, 
                 extras_size):
        super(CoMappingRolloutBuffer, self).__init__(rollouts,
                                                  num_envs, 
                                                  eval_freq, 
                                                  num_repeats, 
                                                  num_agents, 
                                                  obs_shape, 
                                                  action_space, 
                                                  rec_state_size)
        self.extras = torch.zeros((rollouts + 1, num_envs * num_agents * num_repeats, extras_size), dtype=torch.long)
        self.has_extras = True
        self.extras_size = extras_size


    def insert(self, 
               obs, 
               rec_states, 
               actions, 
               action_log_probs, 
               value_preds,
               rewards, 
               masks, 
               extras):
        l, h = self.mini_step * self.mini_step_size, (self.mini_step + 1) * self.mini_step_size
        if self.step == 0:
            ll, lh = l-self.mini_step_size, h-self.mini_step_size
            if lh == 0:
                lh = self.mini_step_size * self.num_rollout_blocks
            self.extras[0][l:h].copy_(self.extras[-1][ll:lh])
        self.extras[self.step + 1][l:h].copy_(extras)
        super(CoMappingRolloutBuffer, self).insert(obs, rec_states, actions,
                                                 action_log_probs, value_preds, rewards, masks)
