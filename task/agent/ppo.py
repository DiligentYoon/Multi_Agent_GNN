# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
import torch
import torch.nn.functional as F
import copy

from typing import Mapping, Optional, Any, Dict, Tuple
from torch.nn import Module

from ..base.agent.agent import Agent

class PPOAgent(Agent):
    def __init__(self, 
                 model: Module, 
                 device: torch.device, 
                 cfg: Optional[dict] = None):
    
        super().__init__(model, device, cfg)

        
        self.optimizer = self.model.optimizer
        # self.actor_optimizer = self.model.actor_optimizer
        # self.critic_optimizer = self.model.critic_optimizer

        self.epoch = self.cfg["epoch"]
        self.mini_batch_size = self.cfg['mini_batch_size']
        self.max_batch_size = self.cfg["max_batch_size"]

        self.rotation_augmentation = self.cfg["rotation_augmentation"]
        
        self.clip_ratio = self.cfg["clip_ratio"]
        self.value_clip_ratio = self.cfg.get("value_clip_ratio", self.clip_ratio)
        self.discount_factor = self.cfg["discount_factor"]
        self.gae_lambda = self.cfg["gae_lambda"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.value_loss_coef = self.cfg["value_loss_coef"]
        self.entropy_loss_coef = self.cfg["entropy_loss_coef"]
        self.policy_loss_coef = self.cfg["policy_loss_coef"]
        

        self.use_clipped_value_loss = True
    
    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        
        return self.model.act(inputs, rnn_hxs, masks, extras, deterministic)

    
    def update(self, data):
        advantages = data.returns[:-1] - data.value_preds[:-1]
        rollouts_begin = (data.mini_step_size * data.num_steps) if data.first_use_to_eval else 0
        valid_advantages = advantages[data.masks[:-1].bool()][rollouts_begin:]
        if min(valid_advantages.shape) == 0:
            print('empty samples ... skip !')
            return 0, 0, 0

        # advantage 표준화
        advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-5)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        approx_kl = 0
        total_norm = 0
        for e in range(self.epoch):
            data_generator = data.sample_mini_batch(advantages, 
                                                    self.mini_batch_size, 
                                                    self.max_batch_size, 
                                                    self.rotation_augmentation, 
                                                    ds=1,
                                                    verbose=True)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                # for i in range(sample['obs'].shape[0]):
                #     if torch.nonzero(sample['obs'][i, 1, :, :]).shape[0] <= torch.max(sample['actions'][i]):
                #         raise ValueError("Action exceeds distribution")
                values, action_log_probs, dist_entropy, _, _ = \
                    self.model.evaluate_actions(
                        sample['obs'], sample['rec_states'],
                        sample['masks'], sample['actions'],
                        extras=sample['extras']
                    )
                
                augmentation = sample['augmentation']
                value_preds = values if augmentation else sample['value_preds']
                old_action_log_probs = action_log_probs if augmentation else sample['old_action_log_probs']
                returns = sample['returns']
                adv_targ = sample['adv_targ']

                ratio = torch.exp(action_log_probs - old_action_log_probs)
                
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                if torch.isnan(action_loss):
                    print('aloss nan')
                    continue
                
                if values.shape != returns.shape:
                    values = values.view(returns.shape)

                if self.use_clipped_value_loss:
                    value_losses = F.mse_loss(values, returns)
                    value_losses_clipped = F.mse_loss((torch.clamp(values, value_preds - self.clip_ratio, value_preds + self.clip_ratio)), returns)
                    value_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = F.mse_loss(values, returns)

                
                self.optimizer.zero_grad()
                # self.actor_optimizer.zero_grad()
                # self.critic_optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss * self.policy_loss_coef - dist_entropy * self.entropy_loss_coef).backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.network.parameters(), self.grad_norm_clip)
                # torch.nn.utils.clip_grad_norm_(self.model.network.actor.parameters(), self.grad_norm_clip)
                # torch.nn.utils.clip_grad_norm_(self.model.network.critic.parameters(), self.grad_norm_clip)

                for name, p in self.model.network.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        print(f"[NaN GRAD] {name} grad has NaN/Inf")
                        raise RuntimeError

                self.optimizer.step()

                if not augmentation:
                    num_frontiers = (sample['obs'][:, 1, :, :] > 0).float().sum(dim=(1, 2))
                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item() / num_frontiers.mean().item()
                
                # Aggregate Gradient Norms
                for p in self.model.network.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item()
                
                # Approximate KL Divergence
                approx_kl += (old_action_log_probs - action_log_probs).mean().item()
        
        for k, v in self.model.network.state_dict().items():
            if torch.isnan(v).any():
                print(f"{k} layer is nan after parameter update")
                raise RuntimeError

        num_updates = self.epoch * self.mini_batch_size

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        approx_kl /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, total_norm, approx_kl