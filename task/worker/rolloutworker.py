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

from collections import deque
from task.env.nav_env import NavEnv
from task.agent.ppo import PPOAgent
from task.buffer.rolloutbuffer import CoMappingRolloutBuffer
from task.model.models import RL_ActorCritic

@ray.remote
class RolloutWorker:
    """
    A Ray Actor that manages an environment instance, collects experience, and
    returns it as a RolloutBuffer. This worker is stateful and can continue
    an episode across multiple calls to `sample()`.
    """
    def __init__(self, worker_id: int, cfg: dict, device: torch.device = torch.device("cpu")):
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
        
        self.worker_id = worker_id
        self.cfg = cfg
        self.device = torch.device("cpu")
        
        # --- Environment, Model, Agent, and Buffer ---
        # These are created once per worker.
        self.env = NavEnv(episode_index=worker_id, device=self.device, cfg=cfg['env'])
        
        pr = self.env.cfg.pooling_downsampling_rate
        num_agents = cfg['env']['num_agent']
        self.observation_space = gym.spaces.Box(0, 1, (8 + num_agents, self.env.map_info.H // pr, self.env.map_info.W // pr), dtype='uint8')
        self.action_space = gym.spaces.Box(0, (self.env.map_info.H // pr) * (self.env.map_info.W // pr) - 1, (num_agents,), dtype='int32')
        
        actor_critic_model = self._create_model(cfg['model'])
        self.agent = PPOAgent(actor_critic_model, self.device, cfg['agent'])
        
        self.rollout_fragment_length = self.cfg['agent']['buffer']['rollout']
        self.rollout_log_interval = 100

        # --- Stateful variables for continuing episodes ---
        self.cumulative_episode_step = deque(maxlen=100)
        self.coverage_rate = deque(maxlen=100)
        self.is_success = deque(maxlen=100)
        self.episode_step = 0
        self.last_rec_states = None
        self.last_mask = None
        self.episode_is_done = True


    def _create_model(self, model_cfg: dict) -> RL_ActorCritic:
        """
        Helper function to create a model instance.
        """
        for key, value in model_cfg.items():
            if key in ['actor_lr', 'critic_lr', 'eps'] and isinstance(value, str):
                model_cfg[key] = float(value)
        return RL_ActorCritic(self.observation_space.shape, self.action_space,
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
            rec_state_size=self.agent.model.rec_state_size,
            extras_size=self.cfg['env']['num_agent'] * 6
        ).to(self.device)


        # Collect a rollout fragment of a fixed length.
        for i in range(self.rollout_fragment_length):
            # If the last episode was done, reset the environment.
            if self.episode_is_done:
                if self.env.is_success:
                    self.is_success.append(1)
                else:
                    self.is_success.append(0)
                self.coverage_rate.append(self.env.prev_explored_region / (self.env.map_info.H * self.env.map_info.W))
                self.cumulative_episode_step.append(copy.deepcopy(self.episode_step))
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

            # print(f"[ Worker {self.worker_id} ] # of frontier at {buffer.step} step : { torch.nonzero(buffer.obs[buffer.step, 0, 1, :, :]).shape[0] }")
            
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

            # if i % self.rollout_log_interval == 0 and i > 0:
            #     print(f"Worker {self.worker_id}: Collected {i} steps of rollout.")
            #     print(f"Mean Rewards : {sum(self.per_step_reward) / len(self.per_step_reward):.2f}")


            # Update state for the next step
            self.last_obs = next_obs
            self.last_info = next_info
            self.last_mask = ~done
            self.last_rec_states = rec_states
            self.episode_is_done = done.item()
            self.episode_step += 1
        
        # --- After the loop, compute the value for the last state ---
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

        episode_step = sum(self.cumulative_episode_step) / len(self.cumulative_episode_step)
        success_rate = sum(self.is_success) / len(self.is_success)
        coverage_rate = sum(self.coverage_rate) / len(self.coverage_rate)

        additional_info = {
            "episode_step": episode_step,
            "success_rate": success_rate,
            "coverage_rate": coverage_rate
        }
        
        # print(f"Worker {self.worker_id}: Finished sampling fragment.")
        return buffer, additional_info


    def set_weights(self, new_weights: dict):
        """
        Updates the model weights of the agent.
        
            Inputs:
                new_weights: A state_dict containing the new weights.
        """
        self.agent.model.network.load_state_dict(new_weights)
