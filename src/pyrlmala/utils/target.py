import json
import os
from typing import Callable, Dict, List, Tuple

import bridgestan as bs
import numpy as np
import numpy.typing as npt
from posteriordb import PosteriorDatabase


class StanTargetPDF:
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
        self.stan_code_path = stan_code_path
        self.posterior_data = posterior_data
        self.model = self.make_model()

    def make_model(self) -> bs.StanModel:
        """
        Create a Stan model from the provided Stan code and posterior data.

        Returns:
            bs.StanModel: The created Stan model.
        """
        stan_data = json.dumps(self.posterior_data)
        return bs.StanModel.from_stan_file(self.stan_code_path, stan_data)

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


class PosteriorDatabaseTargetPDF:
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str = os.path.join("posteriordb", "posterior_database"),
    ) -> None:
        """
        Initializes the PosteriorDatabaseTargetPDF class.
        Args:
            model_name (str): Model name.
            posteriordb_path (str): Path to the database.
        """
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.stan_target = self.generate_model()

    def generate_model(self) -> bs.StanModel:
        """
        Generate a Stan model from the given model name.

        Returns:
            bs.StanModel: The created Stan model.
        """
        # Load DataBase Locally
        pdb = PosteriorDatabase(self.posteriordb_path)

        ## Load Dataset
        posterior = pdb.posterior(self.model_name)
        stan_code_path = posterior.model.stan_code_file_path()
        stan_data = posterior.data.values()

        stan_target = StanTargetPDF(stan_code_path, stan_data)

        return stan_target.model

    def log_target_pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Computes the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Log target pdf values.
        """
        return self.stan_target.log_density(x)

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
        return self.stan_target.log_density_gradient(x)[1]

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
        return self.stan_target.log_density_hessian(x)[2]


class AutoStanTargetPDF:
    """
    AutoStanTargetPDF class for handling Stan models and their associated target probability density functions.
    """

    def __init__(
        self,
        /,
        stan_model: str,
        data: Dict[str, float | int] | str,
    ):
        """
        Initializes the AutoStanTargetPDF class.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.
        """
        self.auto_allocate(stan_model, data)

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

    def auto_allocate(
        self, stan_model: str, data: Dict[str, float | int] | str
    ) -> None:
        """
        Automatically allocate the Stan model and data based on the provided inputs.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.

        Raises:
            ValueError: If stan_model is not a path to a .stan file or a model name, and data is not a dictionary or a path to the posterior database.
        """
        if self.is_existing_stan_file(stan_model) and isinstance(data, dict):
            self.stan_model_path = stan_model
            self.stan_data = data
            self.mode = "stan_file"
        elif isinstance(data, str):
            self.stan_model_name = stan_model
            self.posteriordb_path = data
            self.mode = "posteriordb"
        else:
            raise ValueError(
                "stan_model should be a path to a .stan file or a model name, and data should be a dictionary or a path to the posterior database."
            )

    def log_target_pdf(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Log target pdf values.
        """
        match self.mode:
            case "stan_file":
                stan_target = StanTargetPDF(self.stan_model_path, self.stan_data)
            case "posteriordb":
                stan_target = PosteriorDatabaseTargetPDF(
                    self.stan_model_name, self.posteriordb_path
                )
            case _:
                raise ValueError("Invalid mode: must be 'stan_file' or 'posteriordb'.")

        return stan_target.log_target_pdf(x)

    def grad_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Gradient of the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Gradient log target pdf values.
        """
        match self.mode:
            case "stan_file":
                stan_target = StanTargetPDF(self.stan_model_path, self.stan_data)
            case "posteriordb":
                stan_target = PosteriorDatabaseTargetPDF(
                    self.stan_model_name, self.posteriordb_path
                )
            case _:
                raise ValueError("Invalid mode: must be 'stan_file' or 'posteriordb'.")

        return stan_target.grad_log_target_pdf(x)

    def hess_log_target_pdf(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Hessian of the log target probability density function.

        Args:
            x (npt.NDArray[np.float64]): Input data.

        Returns:
            npt.NDArray[np.float64]: Hessian log target pdf values.
        """
        match self.mode:
            case "stan_file":
                stan_target = StanTargetPDF(self.stan_model_path, self.stan_data)
            case "posteriordb":
                stan_target = PosteriorDatabaseTargetPDF(
                    self.stan_model_name, self.posteriordb_path
                )
            case _:
                raise ValueError("Invalid mode: must be 'stan_file' or 'posteriordb'.")

        return stan_target.hess_log_target_pdf(x)

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
        if not isinstance(mode, list):
            raise ValueError("mode should be a list of strings")
        if not all(i in ["pdf", "grad", "hess"] for i in mode):
            raise ValueError('mode should be one of "pdf", "grad", "hess"')
        if len(mode) == 0:
            raise ValueError("mode should not be empty")

        match self.mode:
            case "stan_file":
                stan_target = StanTargetPDF(self.stan_model_path, self.stan_data)
            case "posteriordb":
                stan_target = PosteriorDatabaseTargetPDF(
                    self.stan_model_name, self.posteriordb_path
                )
            case _:
                raise ValueError("Invalid mode: must be 'stan_file' or 'posteriordb'.")

        funcs = []

        for j in mode:
            match j:
                case "pdf":
                    funcs.append(stan_target.log_target_pdf)
                case "grad":
                    funcs.append(stan_target.grad_log_target_pdf)
                case "hess":
                    funcs.append(stan_target.hess_log_target_pdf)
                case _:
                    raise ValueError('mode should be one of "pdf", "grad", "hess"')
        return tuple(funcs)
