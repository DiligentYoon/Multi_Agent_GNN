# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
import torch
import torch.nn.functional as F

from typing import Mapping, Optional, Any, Dict, Tuple
from torch.nn import Module

from ..base.agent.agent import Agent

class PPOAgent(Agent):
    def __init__(self, 
                 model: Module, 
                 device: torch.device, 
                 cfg: Optional[dict] = None):
    
        super().__init__(model, device, cfg)

        
        self.actor_optimizer = self.model.actor_optimizer
        self.critic_optimizer = self.model.critic_optimizer

        self.epoch = self.cfg["epoch"]
        self.mini_batch_size = self.cfg['mini_batch_size']
        self.max_batch_size = self.cfg["max_batch_size"]

        self.rotation_augmentation = self.cfg["rotation_augmentation"]
        
        self.clip_ratio = self.cfg["clip_ratio"]
        self.discount_factor = self.cfg["discount_factor"]
        self.gae_lambda = self.cfg["gae_lambda"]

        self.grad_norm_clip = self.cfg["grad_norm_clip"]
        self.value_loss_coef = self.cfg["value_loss_coef"]
        self.entropy_loss_coef = self.cfg["entropy_loss_coef"]
        self.policy_loss_coef = self.cfg["policy_loss_coef"]

    
    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        
        return self.model.act(inputs, rnn_hxs, masks, extras, deterministic)

    
    def update(self, data):
        advantages = data.returns[:-1] - data.value_preds[:-1]
        rollouts_begin = (data.mini_step_size * data.num_steps) if data.first_use_to_eval else 0
        valid_advantages = advantages[data.masks[:-1].bool()][rollouts_begin:]
        if min(valid_advantages.shape) == 0:
            print('empty samples ... skip !')
            return 0, 0, 0
        print('min/max/mean/med adv: {:.3f}/{:.3f}/{:.3f}/{:.3f}'.format(valid_advantages.min(), valid_advantages.max(), valid_advantages.mean(), valid_advantages.median()))
        advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-6)
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = data.sample_mini_batch(advantages, 
                                                    self.mini_batch_size, 
                                                    self.max_batch_size, 
                                                    self.rotation_augmentation, 
                                                    ds=self.actor_critic.network.downscaling)

            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, action_feature = \
                    self.actor_critic.evaluate_actions(
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
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()
                action_clip = (torch.abs(ratio - 1.0) <= self.clip_param).float().mean().item()
                if torch.isnan(action_loss):
                    print('aloss nan')
                    continue

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds + \
                                        (values - value_preds).clamp(
                                            -self.clip_param, self.clip_param)
                    value_losses = (values - returns).pow(2)
                    value_losses_clipped = (value_pred_clipped
                                            - returns).pow(2)
                    value_loss = .5 * torch.max(value_losses,
                                                value_losses_clipped).mean()
                    value_clip = (torch.abs(values - value_preds) <= self.clip_param).float().mean().item()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()
                    value_clip = 1.

                print('a/v clip: {:.3f}/{:.3f}'.format(action_clip, value_clip))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()

                (value_loss * self.value_loss_coef + action_loss * self.policy_loss_coef - dist_entropy * self.entropy_loss_coef).backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor_critic.network.actor.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.network.critic.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                if not augmentation:
                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += dist_entropy.item()
        
        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch