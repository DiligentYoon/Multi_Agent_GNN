

class EnvCfg():
    seed: int
    physics_dt: float
    device: str
    num_agent: int
    num_rays: int
    max_velocity: float
    max_yaw_rate: float
    max_acceleration: float
    fov: int
    sensor_range: float
    obs_manager: dict
    controller: dict
    graph: dict
    map: dict

    def __init__(self, cfg: dict) -> None:
        for key, value in cfg.items():
            setattr(self, key, value)
