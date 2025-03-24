from typing import List, Type, TypeAliasType, TypeVar, TYPE_CHECKING

import numpy as np
import torch
from jaxtyping import Float
from numpy import typing as npt

if TYPE_CHECKING:
    from ..envs import BarkerEnv, BarkerESJDEnv, MALAEnv, MALAESJDEnv

ArrayLikeInput = TypeAliasType(
    "ArrayLikeInput",
    npt.NDArray[np.floating]
    | Float[torch.Tensor, "input"]
    | List[float | np.floating | Float[torch.Tensor, "float"]],
)
ArrayLikeOutput = TypeAliasType(
    "ArrayLikeOutput", npt.NDArray[np.floating] | Float[torch.Tensor, "output"]
)

T_x = TypeVar("T_x", bound=ArrayLikeInput, covariant=True)
T_y = TypeVar("T_y", bound=ArrayLikeInput, covariant=True)
T_out = TypeVar("T_out", bound=ArrayLikeOutput, covariant=True)

EnvType = TypeAliasType(
    "EnvType", "Type[BarkerEnv] | Type[BarkerESJDEnv] | Type[MALAEnv] | Type[MALAESJDEnv]"
)
EnvInstanceType = TypeAliasType(
    "EnvInstanceType", "BarkerEnv | BarkerESJDEnv | MALAEnv | MALAESJDEnv"
)
