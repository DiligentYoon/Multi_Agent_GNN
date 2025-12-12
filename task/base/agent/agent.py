from abc import abstractmethod
import os
import torch
from typing import Mapping, Optional
from torch.nn import Module

class Agent:
    """
    Base Multi-Agent class for Centralized Training and Decentralized Execution (CTDE).

    This class lives in the main driver process. It owns the models and defines the
    core logic for action selection and training updates. It does NOT perform
    any direct file I/O or logging.
    """
    def __init__(self,
                 model: Module,
                 device: torch.device,
                 cfg: Optional[dict] = None):
        
        self.model = model
        self.cfg = cfg if cfg is not None else {}
        self.device = device
    
    def set_running_mode(self, mode: str):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()

    def load(self, path, device):
        # Re-initialize optimizers before loading their state
        state_dict = torch.load(path, map_location=device)
        self.model.network.load_state_dict(state_dict['network'])
        self.model.optimizer.load_state_dict(state_dict['optimizer'])
        # self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        # self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        del state_dict
    
    def save(self, path):
        # state = {
        #     'network': self.network.state_dict(),
        #     'actor_optimizer': self.actor_optimizer.state_dict(),
        #     'critic_optimizer': self.critic_optimizer.state_dict(),
        # }
        state = {
            'network': self.model.network.state_dict(),
            'optimizer': self.model.optimizer.state_dict(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    @abstractmethod
    def act(self, states: torch.Tensor) -> torch.Tensor:
        """
        Selects actions for all agents based on their states.
        This method will be used by workers for decentralized execution.

        :param states: A tensor of shape (num_agents, obs_dim)
        :return: A tensor of shape (num_agents, action_dim)
        """
        raise NotImplementedError(f"Please implement the 'act' method for {self.__class__.__name__}.")
    
    @abstractmethod
    def update(self, data, timestep, timesteps) -> None:
        """
        Performs a centralized training update step using a batch of data.
        This method is called by the main driver.

        :param batch: A batch of experience data collected from workers.
        :return: A dictionary containing training statistics (e.g., loss values).
        """
        raise NotImplementedError(f"Please implement the 'update' method for {self.__class__.__name__}.")