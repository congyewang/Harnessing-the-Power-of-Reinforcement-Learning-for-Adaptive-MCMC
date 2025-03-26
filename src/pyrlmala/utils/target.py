import json
import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, TypeGuard

import bridgestan as bs
import numpy as np
import numpy.typing as npt
from posteriordb import PosteriorDatabase


class TargetPDFBase(ABC):
    def __init__(self, stan_model: str, data: str | Dict[str, int | float]) -> None:
        """
        Initializes the TargetPDFBase class.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (str | Dict[str, int  |  float]): Input data for the model.
        """
        self.stan_model = stan_model
        self.data = data
        self.model = self.make_model()

    @abstractmethod
    def make_model(self) -> bs.StanModel:
        """
        Create a Stan model from the provided Stan code and posterior data.

        Returns:
            bs.StanModel: The created Stan model.
        """
        raise NotImplementedError("Method not implemented")

    def log_target_pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Computes the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Log target pdf values.
        """
        return self.model.log_density(x)

    def grad_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes the gradient of the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Gradient log target pdf values.
        """
        return self.model.log_density_gradient(x)[1]

    def hess_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Computes the Hessian of the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Hessian log target pdf values.
        """
        return self.model.log_density_hessian(x)[2]

    def combine_make_log_target_pdf(
        self,
        mode: List[str] = ["pdf", "grad", "hess"],
    ) -> Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ...]:
        """
        Combine the log target pdf, gradient, and hessian functions into a tuple.

        Args:
            mode (List[str], optional): List of modes to include. Defaults to ["pdf", "grad", "hess"].

        Raises:
            ValueError: If the mode is not one of "pdf", "grad", or "hess".
            ValueError: If the mode is not "stan_file" or "posteriordb".

        Returns:
            Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ...]:
        """
        if not all(i in ["pdf", "grad", "hess"] for i in mode):
            raise ValueError('mode should be one of "pdf", "grad", "hess"')
        if len(mode) == 0:
            raise ValueError("mode should not be empty")

        funcs: List[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = []

        for j in mode:
            match j:
                case "pdf":
                    funcs.append(self.model.log_density)
                case "grad":
                    funcs.append(lambda x: self.model.log_density_gradient(x)[1])
                case "hess":
                    funcs.append(lambda x: self.model.log_density_hessian(x)[2])
                case _:
                    raise ValueError('mode should be one of "pdf", "grad", "hess"')

        return tuple(funcs)


class StanTargetPDF(TargetPDFBase):
    """
    StanTargetPDF class for handling Stan models and their associated target probability density functions.
    """

    def __init__(
        self, stan_code_path: str, posterior_data: Dict[str, float | int]
    ) -> None:
        """
        Initializes the StanTargetPDF class.

        Args:
            stan_code_path (str): Path to the Stan code file.
            posterior_data (Dict[str, float | int]): Posterior data for the model.
        """
        super().__init__(stan_code_path, posterior_data)

    def make_model(self) -> bs.StanModel:
        """
        Create a Stan model from the provided Stan code and posterior data.

        Returns:
            bs.StanModel: The created Stan model.
        """
        stan_data = json.dumps(self.data)
        return bs.StanModel.from_stan_file(self.stan_model, stan_data)


class PosteriorDatabaseTargetPDF(TargetPDFBase):
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str,
    ) -> None:
        """
        Initializes the PosteriorDatabaseTargetPDF class.

        Args:
            model_name (str): Model name.
            posteriordb_path (str): Path to the database.
        """
        super().__init__(model_name, posteriordb_path)

    @staticmethod
    def is_posteriordb_path(path: str | Dict[str, float | int]) -> TypeGuard[str]:
        """
        Check if the given path is a valid posteriordb path.
        Args:
            path (str | Dict[str, float | int]): Path to the database.
        Returns:
            TypeGuard[str]: True if the path is a valid posteriordb path, False otherwise.
        """
        return isinstance(path, str) and "posterior_database" in path

    def make_model(self) -> bs.StanModel:
        """
        Generate a Stan model from the given model name.

        Returns:
            bs.StanModel: The created Stan model.

        Raises:
            ValueError: If the path is not a dictionary.
        """
        if self.is_posteriordb_path(self.data):
            # Load DataBase Locally
            pdb = PosteriorDatabase(self.data)

            ## Load Dataset
            posterior = pdb.posterior(self.stan_model)
            stan_code_path = posterior.model.stan_code_file_path()
            stan_data = posterior.data.values()

            stan_target = StanTargetPDF(stan_code_path, stan_data)

            return stan_target.model
        else:
            raise ValueError(
                "posteriordb_path should be a path to the database, not a dictionary"
            )


class AutoStanTargetPDF(TargetPDFBase):
    """
    AutoStanTargetPDF class for handling Stan models and their associated target probability density functions.
    """

    def __init__(
        self,
        /,
        stan_model: str,
        data: Dict[str, float | int] | str,
    ) -> None:
        """
        Initializes the AutoStanTargetPDF class.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.
        """
        super().__init__(stan_model, data)

    @staticmethod
    def is_existing_stan_file(path: str) -> bool:
        """
        Check if the given path is an existing Stan file.

        Args:
            path (str): Path to the Stan file.

        Returns:
            bool: True if the file exists and is a Stan file, False otherwise.
        """
        return os.path.isfile(path) and path.endswith(".stan")

    def make_model(self) -> bs.StanModel:
        """
        Create a Stan model based on the provided mode.

        Returns:
            bs.StanModel: The created Stan model.
        """
        if self.is_existing_stan_file(self.stan_model) and isinstance(self.data, dict):
            stan_target = StanTargetPDF(self.stan_model, self.data)

            return stan_target.model

        elif PosteriorDatabaseTargetPDF.is_posteriordb_path(self.data):
            stan_target = PosteriorDatabaseTargetPDF(self.stan_model, self.data)

            return stan_target.model
        else:
            raise ValueError(
                "stan_model should be a path to a .stan file or a model name, and data should be a dictionary or a path to the posterior database."
            )
