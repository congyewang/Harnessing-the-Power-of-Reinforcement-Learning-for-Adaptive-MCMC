from typing import TypeAlias

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.registration import register

from ._env import RLMALAEnv

RLMCMCEnv: TypeAlias = gym.Env[npt.NDArray[np.float64], npt.NDArray[np.float64]]

register(
    id="RLMALAEnv-v1.0",
    entry_point="src.pyrlmala.envs._env:RLMALAEnv",
)

__all__ = ["RLMALAEnv", "RLMCMCEnv"]
