import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
if sys.platform == "win32":
    try:
        import contextlib
        import diffcp.cone_program as dc
        def _dummy_threadpool_limits(*args, **kwargs):
            return contextlib.nullcontext()
        dc.threadpool_limits = _dummy_threadpool_limits
    except Exception:
        pass

import ray
import torch
import yaml
import datetime
import numpy as np
import collections
import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter

from collections import deque
from typing import List
from task.env.nav_env import NavEnv
from task.agent.sac import SACAgent
from task.worker.rolloutworker import RolloutWorker
from task.agent.ppo import Agent, PPOAgent
from task.model.models import RL_ActorCritic
from task.buffer.rolloutbuffer import RolloutBuffer, CoMappingRolloutBuffer



class MainDriver:
    """
    The main orchestrator for the training process.
    It manages worker creation, data collection, centralized training,
    logging, and checkpointing.
    """
    def __init__(self, cfg: dict):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=1)
        except Exception:
            pass
        self.cfg = cfg
        self.timesteps = self.cfg["train"]["timesteps"]
        self.start_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        # --- Ray Processor ---
        ray.init(num_cpus=self.cfg['ray']['num_cpus'])
        print(f"Ray initialized with {self.cfg['ray']['num_cpus']} CPUs.")
        # --- Device ---
        self.device = torch.device(self.cfg['env']['device'])
        # --- Experiment Directory and Logging ---
        self.experiment_dir = os.path.join("results", f"{self.start_time}_{self.cfg['agent']['experiment']['directory']}")
        self.writer = SummaryWriter(log_dir=self.experiment_dir)
        self.write_interval = self.cfg['agent']['experiment']['write_interval']
        if self.write_interval == 'auto':
            self.write_interval = int(self.timesteps / 30)

        self.checkpoint_interval = self.cfg['agent']['experiment']['checkpoint_interval']
        if self.checkpoint_interval == 'auto':
            self.checkpoint_interval = int(self.timesteps / 10)

        self.cumulative_metrics = {}
        print(f"TensorBoard logs will be saved to: {self.experiment_dir}")

        # --- Environment Info & Model and Agent & Buffer ---
        temp_env = NavEnv(episode_index=0, device=self.device, cfg=self.cfg['env'])
        num_agents = cfg['env']['num_agent']
        pr = temp_env.cfg.pooling_downsampling_rate
        observation_space = gym.spaces.Box(0, 1, (8 + num_agents, temp_env.map_info.H // pr, temp_env.map_info.W // pr), dtype='uint8')
        action_space = gym.spaces.Box(0, (temp_env.map_info.H // pr) * (temp_env.map_info.W // pr) - 1, (num_agents,), dtype='int32')
        actor_critic_model      = self.create_model(cfg['model'], observation_space, action_space, self.device)
        self.master_agent, self.buffer = self.create_agent(cfg['agent'], actor_critic_model, num_agents,
                                                           cfg['train']['eval_freq'], observation_space, action_space, self.device)
        del temp_env

        print("[INFO] Master Agent and Buffer created.")

        # --- Worker Creation for Parallel Working ---
        self.workers = [RolloutWorker.remote(worker_id=i, 
                                             env_cfg=self.cfg['env'], 
                                             agent_cfg=self.cfg['agent'],
                                             model_cfg=self.cfg['model']) for i in range(self.cfg['ray']['num_workers'])]
        
        print(f"{self.cfg['ray']['num_workers']} Workers created.")

        # --- Data Logging ---
        self.tracking_data = collections.defaultdict(list)
        self._track_episode_rewards = collections.deque(maxlen=50)
        self._track_episode_lengths = collections.deque(maxlen=50)
        self._track_instantaneous_rewards = collections.deque(maxlen=50)

        # --- Curriculum ---
        self.curriculum_variables = self.cfg['env'].get("curriculum", {})



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


    def train(self):
        """Main training loop."""
        print("=== Training Start ===")

        current_policy_weights = self.master_agent.model.get_policy_data()
        # Broadcast initial weights to all workers
        cpu_policy_weights = {k: v.cpu() for k, v in current_policy_weights.items()}
        for worker in self.workers:
            worker.set_weights.remote(cpu_policy_weights, role="policy")

        # Start the first batch of rollouts
        jobs = [worker.rollout.remote(i, self.curriculum_variables) for i, worker in enumerate(self.workers)]
        self.global_episode_count = len(self.workers)
        self.max_episode = self.cfg["train"]["max_episode"]

        global_step = 0
        while global_step < self.cfg['train']['timesteps']:
            # Wait for any worker to finish a rollout
            done_ids, jobs = ray.wait(jobs)

            # Process results from all completed workers
            for done_id in done_ids:
                loss_dict = {} # Initialize loss_dict for logging
                result = ray.get(done_id)
                worker_id = result['worker_id']
                metrics = result['metrics']
                trajectory = result['trajectory']

                # Add collected data to the replay buffer
                for transition in trajectory:
                    self.replay_buffer.push(transition)
                
                # Update Episode Id
                episode_length = metrics.get('episode_length', 0)
                episode_reward = metrics.get('episode_reward', 0)
                global_step += episode_length

                # Tracking Episode-wise Data from ray Worker
                self._track_data(metrics)

                # --- Centralized Training Step ---
                if len(self.replay_buffer) >= self.master_agent.minimum_buffer_size:
                    critic_loss = 0
                    policy_loss = 0
                    for _ in range(self.cfg['agent']['gradient_steps']):
                        batch = self.replay_buffer.sample(self.cfg['agent']['batch_size'])
                        loss_dict = self.master_agent.update(batch)
                        self._track_data(loss_dict)
                        critic_loss += loss_dict.get('critic_loss', 0)
                        policy_loss += loss_dict.get('policy_loss', 0)
                    
                # Update weights to be sent to workers
                current_weights = self.master_agent.get_checkpoint_data()
                current_policy_weights = current_weights['policy']
                currnet_policy_feature_extractor_weights = current_weights["policy_feature_extractor"]
                cpu_policy_weights = {k: v.cpu() for k, v in current_policy_weights.items()}
                cpu_policy_feature_extractor_weights = {k: v.cpu() for k, v in currnet_policy_feature_extractor_weights.items()}

                # --- Logging & Save Checkpoint ---
                # Log
                if global_step > 0 and (global_step - episode_length) // self.write_interval < global_step // self.write_interval:
                    self._write_tracking_data(global_step)
                # Checkpoint
                if global_step > 0 and (global_step - episode_length) // self.checkpoint_interval < global_step // self.checkpoint_interval:
                    self._save_checkpoint(global_step)

                # --- Print Metrics to Console ---
                log_message = (
                    f"Steps: {global_step:<7} | "
                    f"Ep Rew: {episode_reward:<8.2f} | "
                    f"Ep Len: {episode_length:<4}"
                )
                if loss_dict:
                    log_message += (
                        f" | Critic Loss: {critic_loss/self.cfg['agent']['gradient_steps']:<7.4f} | "
                        f"Policy Loss: {policy_loss/self.cfg['agent']['gradient_steps']:<7.4f}"
                    )
                print(log_message)

                # --- Update Scheduled Variables
                self.scheduling_by_timestep(global_step)

                print("Scheduling Variables: ")
                print(f"1. demo_rate: {self.curriculum_variables['demo'].get('demo_rate'):<7.4f}")

                # --- Launch New Job for the finished worker ---
                self.workers[worker_id].set_weights.remote(cpu_policy_weights, role="policy")
                self.workers[worker_id].set_weights.remote(cpu_policy_feature_extractor_weights, role="policy_feature_extractor")
                new_job = self.workers[worker_id].rollout.remote(self.global_episode_count % self.max_episode, self.curriculum_variables)
                self.global_episode_count += 1
                jobs.append(new_job)

        print("\n=== Training Finished ===")
        ray.shutdown()


    def scheduling_by_timestep(self, timestep: int):
        progress = timestep / self.cfg['train']['timesteps']

        # Demonstrations
        demo_dict = self.curriculum_variables['demo']
        min_demo_rate = demo_dict['end_rate']
        max_demo_rate = demo_dict['start_rate']
        self.curriculum_variables['demo']['demo_rate'] = max(min_demo_rate, max_demo_rate-progress)



    def scheduling_by_performance(self):
        pass


    def _track_data(self, data: dict):
        for key, value in data.items():
            if "episode_reward_team" in key:
                self._track_episode_rewards.append(value)
            elif "episode_length" in key:
                self._track_episode_lengths.append(value)
            elif "instantaneuous_reward_team" in key:
                self._track_instantaneous_rewards.append(value)
            else:
                self.tracking_data[key].append(value)


    def _write_tracking_data(self, global_step: int):
        if len(self._track_episode_rewards) > 0:
            rewards_arr = np.array(self._track_episode_rewards)
            lengths_arr = np.array(self._track_episode_lengths)
            rewards_arr_i = np.array(self._track_instantaneous_rewards)
            
            self.writer.add_scalar("Reward/Total Team reward (mean)", np.mean(rewards_arr), global_step)
            self.writer.add_scalar("Reward/Total Team reward (max)", np.max(rewards_arr), global_step)
            self.writer.add_scalar("Reward/Total Team reward (min)", np.min(rewards_arr), global_step)

            self.writer.add_scalar("Reward/Total Team instantaneous reward (mean)", np.mean(rewards_arr_i), global_step)
            self.writer.add_scalar("Reward/Total Team instantaneous reward (max)", np.max(rewards_arr_i), global_step)
            self.writer.add_scalar("Reward/Total Team instantaneous reward (min)", np.min(rewards_arr_i), global_step)
            
            self.writer.add_scalar("Episode/Total timesteps (mean)", np.mean(lengths_arr), global_step)
            self.writer.add_scalar("Episode/Total timesteps (max)", np.max(lengths_arr), global_step)
            self.writer.add_scalar("Episode/Total timesteps (min)", np.min(lengths_arr), global_step)
        
        for key, values in self.tracking_data.items():
            self.writer.add_scalar(key, np.mean(values), global_step)
        
        self.tracking_data.clear()
        self.writer.flush()



    def _save_checkpoint(self, global_step: int):
        """Saves a checkpoint of the master agent's models."""
        filepath = os.path.join(self.experiment_dir, "checkpoints", f"agent_{global_step}.pt")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.master_agent.get_checkpoint_data(), filepath)
        print(f"--- Checkpoint saved at step {global_step} ---")



if __name__ == '__main__':
    with open("config/nav_ppo_cfg.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    driver = MainDriver(cfg=config)
    driver.train()

