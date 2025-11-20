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

from torch.utils.tensorboard import SummaryWriter

from task.env.cbf_env import CBFEnv
from task.agent.sac import SACAgent
from task.models.models import ActorGaussianNet, CriticDeterministicNet, GNN_Feature_Extractor, DifferentiableCBFLayer
from task.models.hgat import HierarchicalGAT
from task.worker.off_policy_worker import OffPolicyWorker
from task.buffer.random_buffer import RandomBuffer



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
        
        ray.init(num_cpus=self.cfg['ray']['num_cpus'])
        print(f"Ray initialized with {self.cfg['ray']['num_cpus']} CPUs.")

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

        # --- Environment Info (for model dimensions) ---
        # Create a temporary env to get observation and action dimensions
        temp_env = CBFEnv(episode_index=0, cfg=self.cfg['env'])
        obs_dim = temp_env.cfg.num_obs
        state_dim = temp_env.cfg.num_state
        action_dim = temp_env.cfg.num_act
        num_agents = self.cfg['env']['num_agent']

        # Setting CBF Constraints Variables from Env
        self.cfg['model']['safety']['a_max'] = temp_env.cfg.max_acceleration
        self.cfg['model']['safety']['w_max'] = temp_env.cfg.max_yaw_rate
        self.cfg['model']['safety']['d_max'] = temp_env.cfg.d_max
        self.cfg['model']['safety']['d_obs'] = temp_env.cfg.d_obs
        self.cfg['model']['safety']['d_safe'] = temp_env.cfg.d_safe
        self.cfg['model']['safety']['max_agents'] = temp_env.cfg.max_agents
        self.cfg['model']['safety']['max_obs'] = temp_env.cfg.max_obs
        del temp_env

        # --- Centralized Components ---
        # 1. Master Agent (holds the master networks and optimizers)
        models = self._create_models(obs_dim, state_dim, action_dim)
        self.master_agent = SACAgent(num_agents=num_agents,
                                     models=models,
                                     device=self.device,
                                     cfg=self.cfg['agent'])
        print("[INFO] Master Agent created.")

        # 2. Replay Buffer
        buffer_size = self.cfg['agent']['buffer']['replay_size']
        self.replay_buffer = RandomBuffer(buffer_size)
        print(f"Replay buffer created with max size {buffer_size}.")

        # --- Worker Creation for Parallel Working ---
        self.workers = [OffPolicyWorker.remote(worker_id=i, 
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



    def _create_models(self, obs_dim, state_dim, action_dim) -> dict:
        """Creates the policy and critic models."""
        model_cfg = self.cfg['model']

        # Feature Extractor
        # local_feature_extractor_v = GNN_Feature_Extractor(state_dim, model_cfg['feature_extractor'], "local")
        # local_feature_extractor_p = GNN_Feature_Extractor(obs_dim, model_cfg['feature_extractor'], "local")
        local_feature_extractor_v = HierarchicalGAT(cfg=model_cfg['feature_extractor'], device=self.device)
        local_feature_extractor_p = HierarchicalGAT(cfg=model_cfg['feature_extractor'], device=self.device)

        # Policy
        actor_type = model_cfg['actor']['type']
        if actor_type == "Gaussian":
            policy_input_dim = model_cfg['feature_extractor']['common']['hidden']
            policy = ActorGaussianNet(policy_input_dim, action_dim, self.device, model_cfg['actor'])
        else:
            ValueError("[INFO] TODO : we should construct the deterministic policy for MADDPG ...")

        # Safety
        safety = DifferentiableCBFLayer(cfg=model_cfg["safety"], device=torch.device('cpu')) # CPU 처리

        # Centralized critic input dimension: state + agent actions
        critic_input_dim = model_cfg['feature_extractor']['common']['hidden'] + action_dim
        critic1 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        critic2 = CriticDeterministicNet(critic_input_dim, 1, self.device, model_cfg['critic'])
        
        return {"policy_feature_extractor": local_feature_extractor_p.to(device=self.device),
                "value_feature_extractor": local_feature_extractor_v.to(device=self.device),
                "policy": policy.to(device=self.device), 
                "safety": safety,
                "value_1": critic1.to(device=self.device), 
                "value_2": critic2.to(device=self.device)}

    def train(self):
        """Main training loop."""
        print("=== Training Start ===")

        current_weights = self.master_agent.get_checkpoint_data()
        current_policy_weights = current_weights['policy']
        currnet_policy_feature_extractor_weights = current_weights["policy_feature_extractor"]
        
        # Broadcast initial weights to all workers
        cpu_policy_weights = {k: v.cpu() for k, v in current_policy_weights.items()}
        cpu_policy_feature_extractor_weights = {k: v.cpu() for k, v in currnet_policy_feature_extractor_weights.items()}
        for worker in self.workers:
            worker.set_weights.remote(cpu_policy_weights, role="policy")
            worker.set_weights.remote(cpu_policy_feature_extractor_weights, role="policy_feature_extractor")

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
    with open("config/cbf_test.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    driver = MainDriver(cfg=config)
    driver.train()

