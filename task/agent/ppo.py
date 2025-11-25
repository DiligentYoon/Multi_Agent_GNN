import copy
import itertools
import torch
import torch.nn.functional as F

from typing import Mapping, Optional, Any, Dict, Tuple
from torch.nn import Module

from ..base.agent.agent import Agent

class PPOAgent(Agent):
    def __init__(self, 
                num_agents: int,
                models: Mapping[str, Module], 
                device: torch.device, 
                cfg: Optional[dict] = None):
    
        super().__init__(models, device, cfg)

        self.num_agent = num_agents
