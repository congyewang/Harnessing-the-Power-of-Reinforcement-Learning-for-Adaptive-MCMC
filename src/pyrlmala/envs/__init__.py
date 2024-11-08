from gymnasium.envs.registration import register

from ._env import MCMCEnvBase, BarkerEnv

register(
    id="BarkerEnv-v1.0",
    entry_point="src.pyrlmala.envs._env:BarkerEnv",
)

__all__ = ["MCMCEnvBase", "BarkerEnv"]
