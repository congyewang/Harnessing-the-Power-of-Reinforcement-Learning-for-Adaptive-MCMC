from gymnasium.envs.registration import register

from .env import BarkerEnv, BarkerESJDEnv, MALAEnv, MALAESJDEnv, MCMCEnvBase

register(
    id="BarkerEnv-v1.0",
    entry_point="pyrlmala.envs.env:BarkerEnv",
)

register(
    id="BarkerESJDEnv-v1.0",
    entry_point="pyrlmala.envs.env:BarkerESJDEnv",
)

register(
    id="MALAEnv-v1.0",
    entry_point="pyrlmala.envs.env:MALAEnv",
)

register(
    id="MALAESJDEnv-v1.0",
    entry_point="pyrlmala.envs.env:MALAESJDEnv",
)

__all__ = ["MCMCEnvBase", "BarkerEnv", "MALAEnv", "BarkerESJDEnv", "MALAESJDEnv"]
