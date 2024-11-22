from gymnasium.envs.registration import register

from .env import BarkerEnv, MCMCEnvBase

register(
    id="BarkerEnv-v1.0",
    entry_point="src.pyrlmala.envs.env:BarkerEnv",
)

__all__ = ["MCMCEnvBase", "BarkerEnv"]
