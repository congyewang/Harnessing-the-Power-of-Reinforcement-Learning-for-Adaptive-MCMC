import glob
import itertools
import json
import os
import re
import tempfile
import warnings
from functools import lru_cache, partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import bridgestan as bs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from cmdstanpy import CmdStanModel
from cytoolz import compose_left, pipe, thread_last
from cytoolz.curried import map, topk
from gymnasium.envs.registration import EnvSpec
from ignite.engine import Engine
from ignite.metrics import MaximumMeanDiscrepancy
from jaxtyping import Float
from matplotlib.axes import Axes
from posteriordb import PosteriorDatabase
from scipy.spatial.distance import pdist
from scipy.stats._multivariate import _PSD
from torch.nn import functional as F


class Toolbox:
    """
    Toolbox class. Contains utility functions. All methods are static.
    """

    @staticmethod
    def make_env(
        env_id: str | EnvSpec,
        log_target_pdf: Callable[
            [float | np.float64 | npt.NDArray[np.float64]],
            float | np.float64,
        ],
        grad_log_target_pdf: Callable[
            [float | np.float64 | npt.NDArray[np.float64]],
            float | np.float64,
        ],
        initial_sample: np.float64 | npt.NDArray[np.float64],
        initial_covariance: Optional[np.float64 | npt.NDArray[np.float64]] = None,
        initial_step_size: float = 1.0,
        total_timesteps: int = 500_000,
        max_steps_per_episode: int = 500,
        log_mode: bool = True,
        seed: int = 42,
    ) -> Callable[[], gym.Env]:
        """
        Make an environment.

        Args:
            env_id (str | EnvSpec): Environment ID.
            log_target_pdf (Callable[[float | np.float64 | npt.NDArray[np.float64]], float | np.float64]): Log target pdf function.
            grad_log_target_pdf (Callable[[float | np.float64 | npt.NDArray[np.float64]], float | np.float64]): Gradient log target pdf function.
            initial_sample (np.float64 | npt.NDArray[np.float64]): Initial sample.
            initial_covariance (Optional[np.float64 | npt.NDArray[np.float64]], optional): Initial covariance. Defaults to None.
            total_timesteps (int, optional): Total timesteps. Defaults to 500_000.
            max_steps_per_episode (int, optional): Max steps per episode. Defaults to 500.
            log_mode (bool, optional): Log mode. Defaults to True.
            seed (int, optional): Seed. Defaults to 42.

        Returns:
            Callable[[], gym.Env]: Environment thunk.
        """

        def thunk() -> gym.Env:
            """
            Create an environment.

            Returns:
                gym.Env: Environment.
            """
            env = gym.make(
                env_id,
                log_target_pdf_unsafe=log_target_pdf,
                grad_log_target_pdf_unsafe=grad_log_target_pdf,
                initial_sample=initial_sample,
                initial_covariance=initial_covariance,
                initial_step_size=initial_step_size,
                total_timesteps=total_timesteps,
                max_steps_per_episode=max_steps_per_episode,
                log_mode=log_mode,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)

            return env

        return thunk

    @classmethod
    def nearestPD(cls, A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
            matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Args:
            A (npt.NDArray[np.float64]): Input array.

        Returns:
            npt.NDArray[np.float64]: Nearest positive-definite matrix.
        """

        B = (A + A.T) / 2
        _, s, V = np.linalg.svd(B)

        H = np.dot(V.T, np.dot(np.diag(s), V))

        A2 = (B + H) / 2

        A3 = (A2 + A2.T) / 2

        if cls.isPD(A3):
            return A3

        spacing = np.spacing(np.linalg.norm(A))
        # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
        # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
        # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
        # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
        # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
        # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
        # `spacing` will, for Gaussian random matrixes of small dimension, be on
        # othe order of 1e-16. In practice, both ways converge, as the unit test
        # below suggests.
        I = np.eye(A.shape[0])
        k = 1
        while not cls.isPD(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1

        return A3

    @classmethod
    def isPD(cls, B: npt.NDArray[np.float64]) -> bool:
        """
        Returns true when input is positive-definite, via Cholesky, det, and _PSD from scipy.

        Args:
            B (npt.NDArray[np.float64]): Input array.

        Returns:
            bool: True if input is positive-definite, False otherwise.
        """
        try:
            _ = np.linalg.cholesky(B)
            res_cholesky = True
        except np.linalg.LinAlgError:
            res_cholesky = False

        try:
            assert np.linalg.det(B) > 0, "Determinant is negative"
            res_det = True
        except AssertionError:
            res_det = False

        try:
            _PSD(B, allow_singular=False)
            res_PSD = True
        except Exception as e:
            if re.search("[Pp]ositive", str(e)):
                return False
            else:
                raise

        res = res_cholesky and res_det and res_PSD

        return res

    @staticmethod
    def make_log_target_pdf(
        stan_code_path: str,
        posterior_data: Dict[str, float | int],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Create a log target pdf function.

        Args:
            stan_code_path (str): Stan code path.
            posterior_data (Dict[str, float  |  int]): Posterior data.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: Log target pdf function.
        """
        stan_data = json.dumps(posterior_data)
        model = bs.StanModel.from_stan_file(stan_code_path, stan_data)

        return model.log_density

    @staticmethod
    def make_grad_log_target_pdf(
        stan_code_path: str,
        posterior_data: Dict[str, float | int],
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Create a gradient log target pdf function.

        Args:
            stan_code_path (str): Stan code path.
            posterior_data (Dict[str, float  |  int]): Posterior data.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: Gradient log target pdf function.
        """
        stan_data = json.dumps(posterior_data)
        model = bs.StanModel.from_stan_file(stan_code_path, stan_data)

        return lambda x: model.log_density_gradient(x)[1]

    @staticmethod
    def create_folder(file_path: str) -> None:
        """
        Create a folder if it does not exist.

        Args:
            file_path (str): File path.
        """
        folder_path = os.path.dirname(file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def imbalanced_mesh_2d(
        x_range: Float[torch.Tensor, "x_range"], y_range: Float[torch.Tensor, "y_range"]
    ) -> Float[torch.Tensor, "mesh_2d"]:
        """
        Construct a 2D meshgrid from x and y ranges.

        Args:
            x_range (Float[torch.Tensor, "x_range"]): x range.
            y_range (Float[torch.Tensor, "y_range"]): y range.

        Returns:
            Float[torch.Tensor, "mesh_2d"]: 2D meshgrid.
        """
        x_repeat = x_range.repeat_interleave(len(y_range))
        y_tile = y_range.repeat(len(x_range))

        return torch.stack([x_repeat, y_tile], dim=1)

    @staticmethod
    def plot_agent(
        indicate: npt.NDArray[np.float64],
        steps_per_episode: int = 100,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the agent's performance.

        Args:
            indicate (npt.NDArray[np.float64]): Indicate array.
            steps_per_episode (int, optional): Steps per episode. Defaults to 100.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        time_points = np.arange(
            steps_per_episode,
            steps_per_episode * (len(indicate) + 1),
            steps_per_episode,
        )

        plt.plot(time_points, indicate)

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def policy_plot_2D_heatmap(
        policy: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        x_range: Float[torch.Tensor, "x"],
        y_range: Float[torch.Tensor, "y"],
        softplus_mode: bool = True,
        save_path: Optional[str] = None,
        title_addition: str = "",
        axes: Optional[Axes] = None,
    ) -> None:
        """
        Plot the policy heatmap.

        Args:
            policy (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Policy function.
            x_range (Float[torch.Tensor, "x"], optional): x range. e.g. torch.arange(-5, 5, 0.1).
            y_range (Float[torch.Tensor, "y"], optional): y range. e.g. torch.arange(-5, 5, 0.1).
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            save_path (Optional[str], optional): Save path. Defaults to None.
            axes (Optional[plt.Axes]): External axes for subplots. If None, creates a new figure.
        """
        if axes is None:
            _, axes = plt.subplots()

        # Plot heatmap
        heatmap_plot = lambda x: axes.imshow(
            x.T,
            extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

        if softplus_mode:
            pipe(
                (x_range, y_range),
                lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                F.softplus,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )
        else:
            pipe(
                (x_range, y_range),
                lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
                lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
                lambda x: x.double(),
                policy,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(len(x_range), len(y_range)),
                heatmap_plot,
            )

        axes.set_title(f"Policy Heatmap {title_addition}")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Action")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def reward_plot(
        reward: npt.NDArray[np.float64],
        step_per_episode: int = 500,
        window_size: int = 5,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the average reward and moving average. Save the plot if save_path is provided.

        Args:
            reward (npt.NDArray[np.float64]): Immediate reward per step.
            step_per_episode (int, optional): Steps per episode. Defaults to 500.
            window_size (int, optional): Window size for moving average. Defaults to 5.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        average_reward = reward.reshape(-1, step_per_episode).mean(axis=1)
        moving_averages = np.convolve(
            a=average_reward, v=np.ones(window_size) / window_size, mode="valid"
        )

        plt.plot(average_reward, label="Average reward")
        plt.plot(moving_averages, label="Moving average")
        plt.legend()
        plt.title("Reward Plot")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def target_plot_1d(
        x_range: Tuple[float, float, int],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            x_range (Tuple[float, float, int]): x range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function for 1D.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        x = np.linspace(*x_range)
        res = np.exp([log_target_pdf(np.array(i, dtype=np.float64)) for i in x])

        plt.plot(x, res)
        plt.title("Target distribution")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def target_plot_2d(
        x_mesh_range: Tuple[float, float, int],
        y_mesh_range: Tuple[float, float, int],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            x_mesh_range (Tuple[float, float, int]): x mesh range.
            y_mesh_range (Tuple[float, float, int]): y mesh range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function for 2D.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(*x_mesh_range), np.linspace(*y_mesh_range)
        )
        x, y = mesh_x.reshape(1, -1), mesh_y.reshape(1, -1)
        data = np.concatenate([x, y], axis=0).T

        res = np.exp(
            np.array([log_target_pdf(np.array(i, dtype=np.float64)) for i in data])
        )

        plt.contourf(mesh_x, mesh_y, res.reshape(x_mesh_range[2], y_mesh_range[2]))
        plt.colorbar()
        plt.title("Target distribution")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def target_plot_multi(
        data_range: Tuple[Tuple[float, float, int], ...],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            data_range (Tuple[Tuple[float, float, int], ...]): Data range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        for i in data_range:
            Toolbox.target_plot_1d(i, log_target_pdf, save_path)

    @staticmethod
    def target_plot(
        data_range: Tuple[Tuple[float, float, int], ...],
        log_target_pdf: Callable[[npt.NDArray[np.float64]], np.float64],
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot the target distribution. Save the plot if save_path is provided.

        Args:
            data_range (Tuple[Tuple[float, float, int], ...]): Data range.
            log_target_pdf (Callable[[npt.NDArray[np.float64]], np.float64]): Log target pdf function.
            save_path (Optional[str], optional): Save path. Defaults to None.
        """
        sample_dim = len(data_range)
        match sample_dim:
            case 1:
                Toolbox.target_plot_1d(data_range[0], log_target_pdf, save_path)
            case 2:
                Toolbox.target_plot_2d(
                    data_range[0], data_range[1], log_target_pdf, save_path
                )
            case _:
                Toolbox.target_plot_multi(data_range, log_target_pdf, save_path)

    @staticmethod
    def detect_environment() -> str:
        """
        Detect the current execution environment.

        Returns:
            str: The detected environment type: 'jupyter', 'ipython', or 'terminal'.
        """
        try:
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":  # Jupyter Notebook or JupyterLab
                return "jupyter"
            elif shell == "TerminalInteractiveShell":  # IPython terminal
                return "ipython"
            else:
                return "terminal"
        except NameError:
            return "terminal"

    @staticmethod
    def get_clear_function() -> Callable[[bool], Any]:
        """
        Get the appropriate clear function based on the environment.

        Returns:
            callable: The function to clear output in the terminal or notebook.
        """
        if Toolbox.detect_environment() == "jupyter":
            from IPython.display import clear_output as clear

            return clear
        else:
            return lambda wait=False: os.system("cls" if os.name == "nt" else "clear")

    @staticmethod
    def inverse_softplus(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Inverse softplus Function for Numerical Stability. y = log(exp(x) - 1).

        Args:
            x (npt.NDArray[np.float64]): Input array.

        Returns:
            npt.NDArray[np.float64]: Inverse softplus of the input with numerical stabilization.
        """
        return x + np.log1p(-np.exp(-x))

    @staticmethod
    def median_trick(gs: npt.NDArray[np.float64]) -> float:
        """
        Compute the median trick.

        Args:
            gs (npt.NDArray[np.float64]): Gold standard array.

        Returns:
            float: Median trick.
        """
        return (0.5 * np.median(pdist(gs))).item()

    @staticmethod
    def gaussian_kernel(
        x: Float[torch.Tensor, "x"], y: Float[torch.Tensor, "y"], sigma: float = 1.0
    ) -> torch.Tensor:
        """
        Compute the RBF (Gaussian) kernel between x and y.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor x.
            y (Float[torch.Tensor, "y"]): Input tensor y.
            sigma (float, optional): Sigma. Defaults to 1.0.

        Returns:
            torch.Tensor: RBF kernel.
        """

        beta = 1.0 / (2.0 * sigma**2)
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-beta * dist_sq)

    @staticmethod
    def batched_mmd(
        x: Float[torch.Tensor, "x"],
        y: Float[torch.Tensor, "y"],
        batch_size: int = 100,
        sigma: float = 1.0,
    ) -> Float[torch.Tensor, "mmd"]:
        """
        Compute the Maximum Mean Discrepancy (MMD) between x and y.

        Args:
            x (Float[torch.Tensor, "x"]): Input tensor x.
            y (Float[torch.Tensor, "y"]): Input tensor y.
            batch_size (int, optional): Batch size. Defaults to 100.
            sigma (float, optional): Sigma. Defaults to 1.0.

        Returns:
            Float[torch.Tensor, "mmd"]: MMD estimate.
        """

        m = x.size(0)
        n = y.size(0)
        mmd_estimate_xx, mmd_estimate_yy, mmd_estimate_xy = 0.0, 0.0, 0.0

        # Compute the MMD estimate in mini-batches
        for i in range(0, m, batch_size):
            x_batch = x[i : i + batch_size]
            for j in range(0, n, batch_size):
                y_batch = y[j : j + batch_size]

                xx_kernel = Toolbox.gaussian_kernel(x_batch, x_batch, sigma)
                yy_kernel = Toolbox.gaussian_kernel(y_batch, y_batch, sigma)
                xy_kernel = Toolbox.gaussian_kernel(x_batch, y_batch, sigma)

                # Compute the MMD estimate for this mini-batch
                mmd_estimate_xx += xx_kernel.sum()
                mmd_estimate_yy += yy_kernel.sum()
                mmd_estimate_xy += xy_kernel.sum()

        # Normalize the MMD estimate
        mmd_estimate = (
            mmd_estimate_xx / m**2
            + mmd_estimate_yy / n**2
            - 2 * mmd_estimate_xy / (m * n)
        )

        return mmd_estimate

    @staticmethod
    def expected_square_jump_distance(data: npt.NDArray[np.float64]) -> np.float64:
        """
        Compute the expected square jump distance.

        Args:
            data (npt.NDArray[np.float64]): Sample data.

        Returns:
            np.float64: Expected square jump distance.
        """
        distances = np.linalg.norm(data[1:] - data[:-1], axis=1)
        return np.mean(distances)

    @staticmethod
    @lru_cache(maxsize=46)
    def gold_standard(
        model_name: str,
        posteriordb_path: str = os.path.join("posteriordb", "posterior_database"),
    ) -> npt.NDArray[np.float64]:
        """
        Generate the gold standard for the given model.

        Args:
            model_name (str): Model name.
            posteriordb_path (str, optional): Path to the database. Defaults to os.path.join("posteriordb", "posterior_database").

        Returns:
            npt.NDArray[np.float64]: Gold standard.
        """
        # Model Preparation
        ## Load DataBase Locally
        pdb_path = os.path.join(posteriordb_path)
        my_pdb = PosteriorDatabase(pdb_path)

        ## Load Dataset
        posterior = my_pdb.posterior(model_name)

        ## Gold Standard
        gs_list = posterior.reference_draws()
        df = pd.DataFrame(gs_list)
        gs_constrain = np.zeros(
            (
                sum(Toolbox.flat(posterior.information["dimensions"].values())),
                posterior.reference_draws_info()["diagnostics"]["ndraws"],
            )
        )
        for i in range(len(df.keys())):
            gs_s: List[str] = []
            for j in range(len(df[df.keys()[i]])):
                gs_s += df[df.keys()[i]][j]
            gs_constrain[i] = gs_s
        gs_constrain = gs_constrain.T

        # Model Generation
        model = Toolbox.generate_model(model_name, posteriordb_path)

        gs_unconstrain = np.array(
            [model.param_unconstrain(np.array(i)) for i in gs_constrain]
        )

        return gs_unconstrain

    @staticmethod
    def flat(nested_list: List[List[Any]]) -> List[Any]:
        """
        Expand nested list
        """
        res = []
        for i in nested_list:
            if isinstance(i, list):
                res.extend(Toolbox.flat(i))
            else:
                res.append(i)
        return res

    @staticmethod
    def generate_model(
        model_name: str,
        posteriordb_path: str = os.path.join("posteriordb", "posterior_database"),
        verbose: bool = False,
    ) -> Callable[[Any], Any]:
        """
        Generate a model from the given model name.

        Args:
            model_name (str): Model name.
            posteriordb_path (str, optional): Path to the database. Defaults to os.path.join("posteriordb", "posterior_database").
            verbose (bool, optional): If True, will print additional information. Defaults to False.

        Returns:
            Callable[[Any], Any]: _description_
        """
        # Load DataBase Locally
        pdb_path = os.path.join(posteriordb_path)
        my_pdb = PosteriorDatabase(pdb_path)

        ## Load Dataset
        posterior = my_pdb.posterior(model_name)
        stan = posterior.model.stan_code_file_path()
        data = json.dumps(posterior.data.values())

        with warnings.catch_warnings():
            if verbose:
                warnings.simplefilter("default", category=UserWarning)
            else:
                warnings.simplefilter("ignore", category=UserWarning)

            model = bs.StanModel.from_stan_file(stan, data)

        return model


class AveragePolicy:
    @staticmethod
    def generate_state_mesh(
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ) -> Callable[
        [Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
        Float[torch.Tensor, "mesh_2d"],
    ]:
        """
        Generate a mesh grid from ranges. The mesh grid is used to evaluate the policy.

        Args:
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Ranges for the mesh grid.

        Returns:
            Callable[[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], Float[torch.Tensor, "mesh_2d"]]: Mesh grid generator.
        """
        return pipe(
            ranges,
            map(lambda r: torch.linspace(*r)),
            lambda ranges: Toolbox.imbalanced_mesh_2d(*ranges),
            lambda x: torch.cat((x, torch.zeros(x.shape)), dim=1),
            lambda x: x.double(),
        )

    @staticmethod
    def calculate_policy(
        weights_path: str,
        actor: Callable[
            [Float[torch.Tensor, "mesh_2d"]], Float[torch.Tensor, "action"]
        ],
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    ) -> Float[torch.Tensor, "action"]:
        """
        Calculate the policy based on the given ranges and weights.

        Args:
            weights_path (str): Path to the model weights.
            actor (Callable[[Float[torch.Tensor, "mesh_2d"]], Float[torch.Tensor, "action"]]): Actor model that produces actions.
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Range for the state mesh.

        Returns:
            Float[torch.Tensor, "action"]: The calculated actions from the policy.
        """
        # Load the weights
        pipe(
            weights_path,
            torch.load,
            actor.load_state_dict,
        )

        return pipe(
            ranges,
            AveragePolicy.generate_state_mesh,
            actor,
        )

    @staticmethod
    def calculate_mean_policy(
        actor: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        weights_path: str,
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        last_step_num: int,
        policy_slice_size: int,
    ) -> Float[torch.Tensor, "average_action"]:
        """
        Calculate the mean policy based on the given weights.

        Args:
            actor (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Actor model that produces actions.
            weights_path (str): Path to the model weights.
            last_step_num (int): Last step number.
            policy_slice_size (int): Policy slice size.
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Range for the state mesh.

        Returns:
            Float[torch.Tensor, "average_action"]: The mean policy
        """
        partial_calculate_policy = partial(
            AveragePolicy.calculate_policy,
            actor=actor,
            ranges=ranges,
        )
        extract_step_number = compose_left(
            lambda x: re.search(r"\d+", x).group(),
            int,
        )
        topk_by_step_num = partial(topk, key=extract_step_number)

        return thread_last(
            weights_path,
            glob.glob,
            topk_by_step_num(last_step_num),
            map(partial_calculate_policy),
            lambda x: itertools.islice(x, 0, None, policy_slice_size),
            list,
            lambda x: torch.stack(x, dim=0),
            lambda x: torch.mean(x, dim=0),
        )

    @staticmethod
    def plot_policy(
        actor: Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]],
        weights_path: str,
        ranges: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        last_step_num: int,
        policy_slice_size: int,
        softplus_mode: bool = True,
        save_path: Optional[str] = None,
        title_addition: str = "",
        axes: Optional[Axes] = None,
    ) -> None:
        """
        Plot the policy based on the provided actor and weights.

        Args:
            actor (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Actor function.
            weights_path (str): Path to the weights.
            last_step_num (int): Last step number to consider for the plot.
            policy_slice_size (int): Number of policies to slice.
            ranges (Tuple[Tuple[float, float, float], Tuple[float, float, float]]): Range for the state mesh.
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
            save_path (Optional[str], optional): Save path. Defaults to None.
            title_addition (str, optional): Title addition for the plot. Defaults to "".
            axes (Optional[plt.Axes]): External axes for subplots. If None, creates a new figure.
        """
        if axes is None:
            _, axes = plt.subplots()

        # Plot heatmap
        heatmap_plot = lambda x: axes.imshow(
            x.T,
            extent=[ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1]],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )

        # Calculate the policy
        if softplus_mode:
            pipe(
                ranges,
                lambda x: AveragePolicy.calculate_mean_policy(
                    actor,
                    weights_path,
                    x,
                    last_step_num,
                    policy_slice_size,
                ),
                F.softplus,
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(ranges[0][2], ranges[1][2]),
                heatmap_plot,
            )
        else:
            pipe(
                ranges,
                lambda x: AveragePolicy.calculate_mean_policy(
                    actor,
                    weights_path,
                    x,
                    last_step_num,
                    policy_slice_size,
                ),
                torch.detach,
                lambda x: x.numpy()[:, 0].reshape(ranges[0][2], ranges[1][2]),
                heatmap_plot,
            )

        axes.set_title(f"Policy Heatmap {title_addition}")
        axes.set_xlabel("x")
        axes.set_ylabel("y")

        cbar = plt.colorbar(axes.images[0], ax=axes, shrink=0.8)
        cbar.set_label("Action")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()


class NUTSFromPosteriorDB:
    """
    NUTS sampler from the posterior database.

    Attributes:
        model_name (str): Model name.
        posteriordb_path (str): Path to the posterior database.
        stan_data (Optional[str]): Stan data.
        model (Optional[CmdStanModel]): CmdStan model.
        temp_file_path (Optional[str]): Temporary file path.
    """

    def __init__(self, model_name: str, posteriordb_path: str) -> None:
        """
        Initialize the NUTS sampler from the posterior database

        Args:
            model_name (str): Model name.
            posteriordb_path (str): Path to the posterior database.
        """
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.stan_data: Optional[str] = None
        self.model: Optional[CmdStanModel] = None
        self.temp_file_path = None

    def load_posterior(self, **kwargs: Any) -> None:
        """
        Load the posterior from the database.

        Args:
            **kwargs (Any): Additional arguments for the CmdStanModel.
        """
        pdb = PosteriorDatabase(self.posteriordb_path)
        posterior = pdb.posterior(self.model_name)
        stan_code = posterior.model.stan_code_file_path()
        self.stan_data = json.dumps(posterior.data.values())

        self.model = CmdStanModel(stan_file=stan_code, **kwargs)

    def output_samples(self, **kwargs: Any) -> npt.NDArray[np.float64]:
        """
        Output the samples from the model.

        Args:
            **kwargs (Any): Additional arguments for the model sampling.

        Returns:
            npt.NDArray[np.float64]: Samples from the model.
        """
        temp_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json")
        self.temp_file_path = temp_file.name

        temp_file.write(self.stan_data)
        temp_file.close()

        fit = self.model.sample(data=self.temp_file_path, **kwargs)

        return fit.stan_variables()

    def close(self) -> None:
        """
        Close the NUTS sampler.
        """
        if os.path.exists(self.temp_file_path):
            os.remove(self.temp_file_path)

    def __enter__(self) -> "NUTSFromPosteriorDB":
        """
        Enter the NUTS sampler.

        Returns:
            NUTSFromPosteriorDB: NUTS sampler.
        """
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """
        Exit the NUTS sampler.

        Args:
            exc_type (Any): Exception type.
            exc_value (Any): Exception value.
            traceback (Any): Traceback.
        """
        self.close()


class CalculateMMD:
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two distributions.
    """

    @staticmethod
    def eval_step(engine, batch):
        return batch

    @staticmethod
    def calculate(
        gs: Float[torch.Tensor, "gold standard"] | npt.NDArray[np.float64],
        accepted_sample: (
            Float[torch.Tensor, "accepted sample"] | npt.NDArray[np.float64]
        ),
        **kwargs: Any,
    ) -> float:
        """
        Calculate the Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            gs (Float[torch.Tensor, &quot;gold standard&quot;] | npt.NDArray[np.float64]): Gold standard.
            accepted_sample (Float[torch.Tensor, &quot;accepted sample&quot;]  |  npt.NDArray[np.float64]): Accepted sample.

        Returns:
            float: Maximum Mean Discrepancy (MMD) between the two distributions.
        """
        default_evaluator = Engine(CalculateMMD.eval_step)

        if len(accepted_sample) > len(gs):
            accepted_sample = accepted_sample[-len(gs) :]

        if not isinstance(gs, torch.Tensor):
            gs = torch.from_numpy(gs)
        if not isinstance(accepted_sample, torch.Tensor):
            accepted_sample = torch.from_numpy(accepted_sample)

        metric = MaximumMeanDiscrepancy(**kwargs)
        metric.attach(default_evaluator, "mmd")
        state = default_evaluator.run([[gs, accepted_sample]])

        return state.metrics["mmd"]
