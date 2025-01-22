import json
from collections.abc import Callable
from typing import Tuple

import numpy as np
import pytest
from crazy_density import CrazyDensity
from numpy import typing as npt

from pyrlmala.utils import Toolbox

DIMS = 2
RANDOM_SEED_LIST = [0, 1, 2, 42, 1234]


class TestCrazyDensity:
    def compile_stan_model(
        self,
    ) -> Tuple[
        Callable[[npt.NDArray[np.float64]], np.float64],
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]:
        """
        Compile the Stan model

        Returns:
            Tuple[Callable[[npt.NDArray[np.float64]], np.float64], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]: The log target pdf and grad log target pdf functions
        """
        model_name = "crazy_density"
        stan_code_path = f"{model_name}.stan"
        stan_data_path = f"{model_name}.json"

        with open(stan_data_path, "r") as f:
            data = json.load(f)

            log_target_pdf = Toolbox.make_log_target_pdf(stan_code_path, data)
            grad_log_target_pdf = Toolbox.make_grad_log_target_pdf(stan_code_path, data)

        return log_target_pdf, grad_log_target_pdf

    @pytest.mark.parametrize("random_seed", RANDOM_SEED_LIST)
    def test_stan_functions(self, random_seed: int) -> None:
        """
        Create a basic CrazyDensity instance for testing.

        Args:
            random_seed (int): The random seed.

        Raises:
            AssertionError: If any of the tests fail.
        """
        log_target_pdf_stan, grad_log_target_pdf_stan = self.compile_stan_model()

        rng = np.random.default_rng(random_seed)
        x = rng.normal(size=(DIMS,))

        assert (
            log_target_pdf_stan(x) == CrazyDensity.log_den_n_mixture(x).item()
        ), "log_target_pdf_np failed, values mismatch"
        assert (
            grad_log_target_pdf_stan(x).shape == x.shape
        ), "grad_log_target_pdf_np failed, shape mismatch"
        assert (
            np.any(np.isnan(grad_log_target_pdf_stan(x))) == False
        ), "grad_log_target_pdf_np failed, NaNs present"
        assert (
            np.any(np.isinf(grad_log_target_pdf_stan(x))) == False
        ), "grad_log_target_pdf_np failed, Infs present"
