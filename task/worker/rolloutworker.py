import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import ray
import numpy as np
import torch
import gymnasium as gym

from typing import Dict, Any

from task.env.nav_env import NavEnv
from task.model.models import RL_ActorCritic
from task.agent.ppo import PPOAgent

@ray.remote
class RolloutWorker:
    """
    A Ray remote actor responsible for collecting experience from the environment.
    It uses a lightweight HAC agent instance to generate actions.
    """
    def __init__(self, 
                 worker_id: int, 
                 env_cfg: Dict[str, Any], 
                 agent_cfg: Dict[str, Any], 
                 model_cfg: Dict[str, Any],
                 device: torch.device):
        """
        Initializes the worker, its environment, and a local lightweight agent.
        It also initializes the persistent state of the environment for rollouts.
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
        self.device = device

        # --- Environment ---
        self.env = NavEnv(episode_index=0, cfg=env_cfg)
        
        # --- Lightweight Agent for Acting ---
        self.agent_cfg = agent_cfg
        
        num_agents = env_cfg['num_agent']
        pr = self.env.cfg.pooling_downsampling_rate
        observation_space = gym.spaces.Box(0, 1, (8 + num_agents, self.env.map_info.H // pr, self.env.map_info.W // pr), dtype='uint8')
        action_space = gym.spaces.Box(0, (self.env.map_info.H // pr) * (self.env.map_info.W // pr) - 1, (num_agents,), dtype='int32')
        actor_critic_model      = self.create_model(model_cfg, observation_space, action_space, self.device)
        self.agent, self.buffer = self.create_agent(agent_cfg, actor_critic_model, num_agents,
                                                    self.cfg['train']['eval_freq'], observation_space, action_space, self.device)


    def create_model(cfg: dict, observation_space: gym.Space, action_space: gym.Space, device: torch.device) -> RL_ActorCritic:
        """
        Helper function to create models based on the config.

            Inputs:
                cfg: Model Configuration Dictionary
                observation_space : observation with respect to specific env
                action_space : action with repsect to specific env
                device : ["cpu", "cuda"]
            
            Returns:
                actor_critic : Actor Critic Model [policy_net, critic_net]
        """
        model_cfg = cfg
        for key, value in model_cfg.items():
            if key in ['actor_lr', 'critic_lr', 'eps'] and isinstance(value, str):
                model_cfg[key] = float(value)

        actor_critic = RL_ActorCritic(observation_space.shape, action_space,
                                    model_type=model_cfg['model_type'],
                                    base_kwargs={'num_gnn_layer': model_cfg['num_gnn_layer'],
                                                'use_history': model_cfg['use_history'],
                                                'ablation': model_cfg['ablation']},
                                    lr=(model_cfg['actor_lr'], model_cfg['critic_lr']),
                                    eps=model_cfg['eps']).to(device)
        return actor_critic


    def create_agent(cfg: dict, model: RL_ActorCritic, num_agents: int, eval_freq: int, 
                    observation_space: gym.Space, action_space: gym.Space, device: torch.device) -> tuple[Agent, RolloutBuffer]:
        """
        Helper function to create agent and corresponding buffer based on the config.

            Inputs:
                cfg: Agent Configuration Dictionary
                models: Actor Critic Model [policy_net, critic_net]
            
            Returns:
                agent : on policy (PPO) or off policy (SAC)
        """
        agent_cfg = cfg
        buffer_cfg = agent_cfg["buffer"]
        algorithm = agent_cfg.get("alg", "PPO")
        if algorithm == "PPO":
            buffer = CoMappingRolloutBuffer(buffer_cfg["rollout"], 
                                            num_envs=1, 
                                            eval_freq=eval_freq,
                                            num_repeats=agent_cfg["num_gae_block"],
                                            num_agents=1, # Centralized Network
                                            obs_shape=observation_space.shape,
                                            action_space=action_space,
                                            rec_state_size=model.rec_state_size,
                                            extras_size=num_agents * 6
                                            ).to(device)
            agent = PPOAgent(model, device, agent_cfg)
        elif algorithm == "SAC":
            raise ValueError("Not Buffer Implementation yet.")
            agent = SACAgent()
        else:
            raise ValueError("Unvalid Algorithm Types.")

        return agent, buffer

    def set_weights(self, policy_weights: Dict[str, Dict[str, torch.Tensor]], role: str = None):
        """
        Updates the local agent's policy networks with new weights from the driver.
        """
        if role == "policy":
            self.agent.policy.load_state_dict(policy_weights)
        else:
            ValueError("Not supported role Type.")

    def rollout(self, episode_index: int, curriculum_variables: Dict) -> Dict[str, Any]:
        """
        Runs one full episode in the environment to collect a trajectory.
        """
        trajectory = []
        obs, state, info = self.env.reset(episode_index=episode_index)
        terminated, truncated = np.zeros((self.env.num_agent, 1), dtype=bool), np.zeros((self.env.num_agent, 1), dtype=bool)
        episode_reward = 0
        episode_length = 0
        self.demo_rate = curriculum_variables['demo']['demo_rate']

        done = False
        while not done:
            # Get actions from the local policy
            with torch.no_grad():
                # Demonstrations by CFVR Based P Controller
                if np.random.randn(1) < self.demo_rate:               
                    demo_nominal = get_nominal_control(p_target=info["nominal"]["p_targets"],
                                                       on_search=info["nominal"]["on_search"],
                                                       v_current=info["safety"]["v_current"],
                                                       a_max=self.env.max_lin_acc,
                                                       w_max=self.env.max_ang_vel,
                                                       v_max=self.env.max_lin_vel)
                    normalized_demo_nominal = torch.tensor(demo_nominal / 
                                                           np.array([self.env.max_lin_acc, 
                                                                     self.env.max_ang_vel])).to(device=self.device)
                    # Demonstration Actions
                    try:
                        actions, Feasible = self.agent.safety(normalized_demo_nominal, info["safety"])
                    except Exception:
                        actions = None
                        Feasible = False
                else:
                    try:
                        # RL Actions
                        _, actions, _, Feasible = self.agent.act(obs, safety_info=info["safety"])
                    except Exception as e:
                        actions = None
                        Feasible = False
            
            if not Feasible:
                print(f"Infeasibility Solver Error.")
                break

            # Step the environment
            next_obs, next_state, rewards, terminated, truncated, next_info = self.env.step(actions)
            
            # Store the complete transition information
            actions_np = actions.detach().cpu().numpy()
            trajectory.append({
                "obs": obs,
                "state": state,
                "info": info,
                "actions": actions_np,
                "rewards": rewards,
                "next_obs": next_obs,
                "next_state": next_state,
                "next_info": next_info,
                "terminated": terminated,
                "truncated": truncated,
            })
            
            obs = next_obs
            state = next_state
            info = next_info
            done = np.any(terminated) | np.any(truncated)
            episode_reward += rewards.sum()
            episode_length += 1

        metrics = {
            f"episode_reward": episode_reward,
            f"episode_length": episode_length,}
        
        return {"trajectory": trajectory, "metrics": metrics, "worker_id": self.worker_id}