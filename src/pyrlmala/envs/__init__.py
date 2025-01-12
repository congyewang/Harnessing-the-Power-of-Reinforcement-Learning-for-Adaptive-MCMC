from gymnasium.envs.registration import register

from .env import BarkerEnv, MALAEnv, MCMCEnvBase

register(
    id="BarkerEnv-v1.0",
    entry_point="pyrlmala.envs.env:BarkerEnv",
)

register(
    id="MALAEnv-v1.0",
    entry_point="pyrlmala.envs.env:MALAEnv",
)

__all__ = ["MCMCEnvBase", "BarkerEnv", "MALAEnv"]
