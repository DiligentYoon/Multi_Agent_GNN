import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ray
import copy
import torch
import random
import gymnasium as gym
import numpy as np

from collections import deque
from task.env.nav_env import NavEnv
from task.agent.ppo import PPOAgent
from task.buffer.rolloutbuffer import CoMappingRolloutBuffer
from task.model.models import RL_ActorCritic
from task.model.models_commaping import RL_CoMapping_Policy
from task.model.models_ver_2 import RL_Policy

@ray.remote
class RolloutWorker:
    """
    A Ray Actor that manages an environment instance, collects experience, and
    returns it as a RolloutBuffer. This worker is stateful and can continue
    an episode across multiple calls to `sample()`.
    """
    def __init__(self,
                 model_version: int,
                 worker_id: int, 
                 cfg: dict,
                 device: torch.device = torch.device("cpu"),
                 map_type: int = 0):
        """
        Initializes the worker.

            Args:
                worker_id: A unique ID for the worker.
                cfg: The configuration dictionary for the environment, model, and agent.
                device: The torch device to run the models on.
        """
        if sys.platform == "win32":
            try:
                import contextlib
                import diffcp.cone_program as dc
                def _dummy_threadpool_limits(*args, **kwargs):
                    return contextlib.nullcontext()

                dc.threadpool_limits = _dummy_threadpool_limits
            except Exception:
                pass
        if map_type == 0:
            cfg['env']['map']['type'] = 'corridor'
        elif map_type == 1:
            cfg['env']['map']['type'] = 'maze'
        elif map_type == 2:
            cfg['env']['map']['type'] = 'random'
        elif map_type == 3:
            cfg['env']['map']['type'] = 'single_maze'
        else:
            cfg['env']['map']['type'] = 'corridor'
        
        self.worker_id = worker_id
        self.cfg = cfg
        self.device = torch.device("cpu")
        
        # --- Seed Setting for Reproducibility ---
        # Set a unique but deterministic seed for each worker
        worker_seed = cfg['env']['seed'] + worker_id
        torch.manual_seed(worker_seed)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        # --- Environment, Model, Agent, and Buffer ---
        self.env = NavEnv(episode_index=worker_id, device=self.device, cfg=cfg['env'])
        
        pr = self.env.cfg.pooling_downsampling_rate
        num_agents = cfg['env']['num_agent']
        self.observation_space = gym.spaces.Box(0, 1, (8 + num_agents, 
                                                       self.env.obs_manager.global_map_size // pr, 
                                                       self.env.obs_manager.global_map_size // pr), dtype='uint8')
        self.action_space = gym.spaces.Box(0, (self.env.obs_manager.global_map_size // pr) * (self.env.obs_manager.global_map_size // pr) - 1, (num_agents,), dtype='int32')
        
        actor_critic_model = self._create_model(cfg['model'], model_version)
        self.agent = PPOAgent(actor_critic_model, self.device, cfg['agent'])
        
        self.rollout_fragment_length = self.cfg['agent']['buffer']['rollout']

        # --- Stateful variables for continuing episodes ---
        self.episode_step = 0
        self.first = True
        self.last_rec_states = None
        self.last_mask = None
        self.episode_is_done = True


    def _create_model(self, model_cfg: dict, model_version: int) -> RL_Policy:
        """
        Helper function to create a model instance.
        """
        for key, value in model_cfg.items():
            if key in ['actor_lr', 'critic_lr', 'eps'] and isinstance(value, str):
                model_cfg[key] = float(value)
        if model_version == 1:
            return RL_ActorCritic(self.observation_space.shape, self.action_space,
                                  model_type=model_cfg['model_type'],
                                  base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                            'use_history': model_cfg['use_history'],
                                            'ablation': model_cfg['ablation']},
                                  lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                  eps=model_cfg['eps']).to(self.device)
        elif model_version == 2:
            return RL_Policy(self.observation_space.shape, self.action_space,
                                model_type=model_cfg['model_type'],
                                base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                            'use_history': model_cfg['use_history'],
                                            'ablation': model_cfg['ablation']},
                                lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                eps=model_cfg['eps']).to(self.device)
        else:
            return RL_CoMapping_Policy(self.observation_space.shape, self.action_space,
                                       model_type=model_cfg['model_type'],
                                       base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                                    'use_history': model_cfg['use_history'],
                                                    'ablation': model_cfg['ablation']},
                                       lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                       eps=model_cfg['eps']).to(self.device)


    def sample(self) -> tuple[CoMappingRolloutBuffer, dict[str, float]]:
        """
        Collects a fragment of experience of `rollout_fragment_length` steps.
        
            Returns:
                A CoMappingRolloutBuffer object containing the collected data.
        """
        # Create a new buffer for this rollout fragment.
        buffer = CoMappingRolloutBuffer(
            self.cfg['agent']['buffer']['rollout'], 
            num_envs=1, 
            eval_freq=self.cfg['train']['eval_freq'],
            num_repeats=self.cfg['agent']["num_gae_block"],
            num_agents=1, # centralized
            obs_shape=self.observation_space.shape,
            action_space=self.action_space,
            rec_state_size=1,
            extras_size=self.cfg['env']['num_agent'] * 6
        ).to(self.device)


        # Collect a rollout fragment of a fixed length.
        is_success = []  
        coverage_rate = []
        episode_step = [] 
        for i in range(self.rollout_fragment_length):
            if self.episode_is_done:
                if self.env.is_success and not self.first:
                    is_success.append(1)
                else:
                    if self.first:
                        self.first = False
                    else:
                        is_success.append(0)
                coverage_rate.append(self.env.prev_explored_region / (self.env.map_info.H * self.env.map_info.W))
                episode_step.append(copy.deepcopy(self.episode_step))
                self.episode_step = 0
                self.last_obs, _, self.last_info = self.env.reset(episode_index=random.randint(0, 100))
            
            # 버퍼에 초기 액션 세팅 + 스텝
            if buffer.step == 0:
                l = buffer.mini_step * buffer.mini_step_size
                h = (buffer.mini_step + 1) * buffer.mini_step_size
                buffer.obs[0][l:h].copy_(self.last_obs.view(1, *self.observation_space.shape))
                buffer.extras[0][l:h].copy_(self.last_info["additional_obs"].view(1, -1) // self.env.cfg.pooling_downsampling_rate)
 
                ll, lh = l-buffer.mini_step_size, h-buffer.mini_step_size
                if lh == 0:
                    lh = buffer.mini_step_size * buffer.num_rollout_blocks
                # Buffer의 Insert에서 이전 롤아웃의 마지막 상태를 현재 롤아웃의 첫 상태로 복사하는 로직에 맞춘 할당 (reset시에만 수행)
                buffer.obs[-1][ll:lh].copy_(buffer.obs[0][l:h])
                buffer.rec_states[-1][ll:lh].copy_(buffer.rec_states[0][l:h])
                buffer.extras[-1][ll:lh].copy_(buffer.extras[0][l:h])

                self.last_rec_states = buffer.rec_states[0][l:h]
                self.last_mask = buffer.masks[0][l:h]
            
            with torch.no_grad():
                values, actions, action_log_probs, \
                rec_states, _ = self.agent.act(
                    self.last_obs.view(1, *self.observation_space.shape),
                    self.last_rec_states,
                    self.last_mask,
                    self.last_info["additional_obs"].view(1, -1) // self.env.cfg.pooling_downsampling_rate,
                    deterministic=False
                )

            next_obs, _, reward, terminated, truncated, next_info = self.env.step(actions)
            done = torch.logical_or(terminated, truncated)

            buffer.insert(
                next_obs.view(1, *self.observation_space.shape), rec_states,
                actions, action_log_probs, values,
                reward, ~done,
                next_info["additional_obs"].view(1, -1) // self.env.cfg.pooling_downsampling_rate
            )

            # Update state for the next step
            self.last_obs = next_obs
            self.last_info = next_info
            self.last_mask = ~done
            self.last_rec_states = rec_states
            self.episode_is_done = done.item()
            self.episode_step += 1
        
        # After the loop, compute the value for the last state
        with torch.no_grad():
            if self.episode_is_done:
                # If the episode ended, the value of the terminal state is 0.
                next_value = torch.zeros(1, 1, device=self.device)
            else:
                # If the episode was cut off, bootstrap from the last observation.
                next_value = self.agent.model.get_value(
                    self.last_obs.view(1, *self.observation_space.shape),
                    self.last_rec_states,
                    self.last_mask, # Should be 1 here as it's not a real 'done'
                    extras=self.last_info["additional_obs"].view(1, -1) // self.env.cfg.pooling_downsampling_rate
                )[0]
        
        buffer.compute_returns(next_value.detach(), True, self.agent.cfg['discount_factor'], self.agent.cfg['gae_lambda'])

        additional_info = {
            "episode_step": episode_step,
            "is_success": is_success,
            "coverage_rate": coverage_rate,
        }
        
        return buffer, additional_info


    def set_weights(self, new_weights: dict):
        """
        Updates the model weights of the agent.
        
            Inputs:
                new_weights: A state_dict containing the new weights.
        """
        self.agent.model.network.load_state_dict(new_weights)
