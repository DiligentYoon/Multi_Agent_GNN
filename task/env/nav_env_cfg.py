
from ..base.env.env_cfg import EnvCfg

class NavEnvCfg(EnvCfg):
    num_obs: int
    num_state: int
    decimation: int
    max_episode_steps: int
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

        # Space Information
        self.num_obs = 1
        self.num_state = 1
        self.num_act = 2

        # Episode Information
        self.decimation = 10
        self.max_episode_steps = 1000

        # Controller Cfg
        self.d_conn = 0.5
        self.d_safe = 0.02
        self.max_obs = self.num_rays
        self.max_agents = self.num_agent

        # Reward Info
        self.reward_weights = {}


