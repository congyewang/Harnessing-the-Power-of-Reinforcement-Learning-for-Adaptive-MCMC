from gymnasium.envs.registration import register

from ._env import RLMALAEnv

register(
    id="RLMALAEnv-v1.0",
    entry_point="src.pyrlmala.envs._env:RLMALAEnv",
)

__all__ = ["RLMALAEnv"]
