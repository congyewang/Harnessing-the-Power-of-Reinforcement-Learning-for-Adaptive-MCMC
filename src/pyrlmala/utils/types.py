from typing import TypeVar, TypeAliasType, List
from jaxtyping import Float
import numpy as np
from numpy import typing as npt
import torch


ArrayLikeInput = TypeAliasType("ArrayLikeInput", npt.NDArray[np.floating] | Float[torch.Tensor, "input"] | List[float | np.floating | Float[torch.Tensor, "float"]])
ArrayLikeOutput = TypeAliasType("ArrayLikeOutput", npt.NDArray[np.floating] | Float[torch.Tensor, "output"])

T_x = TypeVar("T_x", bound=ArrayLikeInput, covariant=True)
T_y = TypeVar("T_y", bound=ArrayLikeInput, covariant=True)
T_out = TypeVar("T_out", bound=ArrayLikeOutput, covariant=True)
