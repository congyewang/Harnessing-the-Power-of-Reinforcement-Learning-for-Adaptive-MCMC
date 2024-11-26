import json
import os
import re
from typing import Callable, Dict, Optional

import bridgestan as bs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.envs.registration import EnvSpec
from jaxtyping import Float
from numpy.typing import NDArray
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
        total_timesteps: int = 500_000,
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
                total_timesteps=total_timesteps,
                log_mode=log_mode,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)

            return env

        return thunk

    @classmethod
    def nearestPD(cls, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Find the nearest positive-definite matrix to input
        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
        credits [2].

        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6

        Args:
            A (NDArray[np.float64]): Input array.

        Returns:
            NDArray[np.float64]: Nearest positive-definite matrix.
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
    def isPD(cls, B: NDArray[np.float64]) -> bool:
        """
        Returns true when input is positive-definite, via Cholesky, det, and _PSD from scipy.

        Args:
            B (NDArray[np.float64]): Input array.

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
    ) -> None:
        """
        Plot the policy heatmap.

        Args:
            policy (Callable[[Float[torch.Tensor, "state"]], Float[torch.Tensor, "action"]]): Policy function.
            x_range (Float[torch.Tensor, "x"], optional): x range. Defaults to torch.arange(-5, 5, 0.1).
            y_range (Float[torch.Tensor, "y"], optional): y range. Defaults to torch.arange(-5, 5, 0.1).
            softplus_mode (bool, optional): Softplus mode. Defaults to True.
        """

        # Plot
        heatmap_plot = lambda x: plt.imshow(
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

        ax = plt.gca()
        ax.set_title("Policy Heatmap")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        plt.colorbar(label="Action")

        if save_path is not None:
            Toolbox.create_folder(save_path)
            plt.savefig(save_path)
        else:
            plt.show()
