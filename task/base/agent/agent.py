from abc import abstractmethod
import torch
from typing import Mapping, Optional
from torch.nn import Module

class Agent:
    """
    Base Multi-Agent class for Centralized Training and Decentralized Execution (CTDE).

    This class lives in the main driver process. It owns the models and defines the
    core logic for action selection and training updates. It does NOT perform
    any direct file I/O or logging.

    Responsibilities:
    1. Holding the master models (actor, critic, etc.).
    2. Providing the `update` method to be called by the driver for centralized training.
    3. Providing the `act` method for inference (can be used by the driver or parts of it, like the policy network, can be shipped to workers).
    """
    def __init__(self,
                 models: Mapping[str, Module],
                 device: torch.device,
                 cfg: Optional[dict] = None):
        
        self.models = models
        self.cfg = cfg if cfg is not None else {}
        self.device = device

        # A dictionary to register modules for checkpointing by the driver
        self.checkpoint_modules = {}

        # Initialize models and move them to the specified device
        for model in self.models.values():
            if model is not None:
                model.to(self.device)
    
    def set_running_mode(self, mode: str):
        if mode == "train":
            print("[INFO] Set Training Mode")
            for model in self.models.values():
                model.train()
        elif mode == "eval":
            print("[INFO] Set Eval Mode")
            for model in self.models.values():
                model.eval()

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
    def update(self, timestep, timesteps) -> None:
        """
            Performs a centralized training update step using a batch of data.
            This method is called by the main driver.

            :param batch: A batch of experience data collected from workers.
            :return: A dictionary containing training statistics (e.g., loss values).
        """
        raise NotImplementedError(f"Please implement the 'update' method for {self.__class__.__name__}.")