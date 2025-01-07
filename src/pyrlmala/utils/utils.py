import json
import os
import re
from typing import Any, Callable, Dict, Optional, Tuple

import bridgestan as bs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.envs.registration import EnvSpec
from jaxtyping import Float
from matplotlib.axes import Axes
from scipy.stats._multivariate import _PSD
from toolz import pipe
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
    def imbalanced_mash_2d(
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
                lambda ranges: Toolbox.imbalanced_mash_2d(*ranges),
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
                lambda ranges: Toolbox.imbalanced_mash_2d(*ranges),
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
        res = np.exp([log_target_pdf(np.array([i], dtype=np.float64)) for i in x])

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
            np.array([log_target_pdf(np.array([i], dtype=np.float64)) for i in data])
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
