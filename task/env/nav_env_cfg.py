
from ..base.env.env_cfg import EnvCfg

class NavEnvCfg(EnvCfg):
    num_obs: int
    num_state: int
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_obs = None # Not use in this env
        self.num_state = None # Not use in this env
        self.num_act = self.num_agent
        self.downsampling_rate = 2
        self.pooling_downsampling_rate = 4

        # Episode Information
        self.decimation = 30
        self.max_episode_steps = 1000

        # Controller Cfg
        self.d_conn = 0.5
        self.d_safe = 0.02
        self.max_obs = self.num_rays
        self.max_agents = self.num_agent

        # Reward Info
        self.reward_weights = {}


