import json
import os
import re
import warnings
from functools import cache
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import bridgestan as bs
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from cytoolz import pipe
from gymnasium.envs.registration import EnvSpec
from jaxtyping import Float
from matplotlib.axes import Axes
from posteriordb import PosteriorDatabase
from torch.nn import functional as F
from loguru import logger
from .mmd import (
    BatchedCalculateMMDTorch,
    CalculateMMDNumpy,
    CalculateMMDTorch,
    MedianTrick,
)
from .nearestpd import NearestPD
from .posteriordb import PosteriorDBToolbox
from .target import AutoStanTargetPDF
from .types import T_x, T_y


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

    @staticmethod
    def nearestPD(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
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

        return NearestPD.nearest_positive_definite(A)

    @staticmethod
    def make_log_target_pdf(
        stan_model: str,
        data: Dict[str, float | int] | str,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Create a log target probability density function from a Stan model.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: Log target pdf function.
        """
        stan_target = AutoStanTargetPDF(stan_model, data)

        return stan_target.log_target_pdf

    @staticmethod
    def make_grad_log_target_pdf(
        stan_model: str,
        data: Dict[str, float | int] | str,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Create a gradient log target probability density function from a Stan model.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: Gradient log target pdf function.
        """
        stan_target = AutoStanTargetPDF(stan_model, data)

        return stan_target.grad_log_target_pdf

    @staticmethod
    def make_hess_log_target_pdf(
        stan_model: str,
        data: Dict[str, float | int] | str,
    ) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
        """
        Create a Hessian log target probability density function from a Stan model.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float  |  int] | str): Posterior data or path to the posterior database.

        Returns:
            Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]: Hessian log target pdf function.
        """
        stan_target = AutoStanTargetPDF(stan_model, data)

        return stan_target.hess_log_target_pdf

    @staticmethod
    def combine_make_log_target_pdf(
        stan_model: str,
        data: Dict[str, float | int] | str,
        mode: List[str] = ["pdf", "grad", "hess"],
    ) -> Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ...]:
        """
        Combine the log target pdf, gradient, and hessian functions into a tuple.

        Args:
            stan_model (str): Path to the Stan model file or model name.
            data (Dict[str, float | int] | str): Posterior data or path to the posterior database.
            mode (List[str], optional): List of modes to include. Defaults to ["pdf", "grad", "hess"].

        Returns:
            Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ...]:
        """
        stan_target = AutoStanTargetPDF(stan_model, data)

        return stan_target.combine_make_log_target_pdf(mode)

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
    def softplus(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Softplus Function for Numerical Stability. y = log(1 + exp(x)).
        This function is used to prevent overflow in the exponentiation.

        Args:
            x (npt.NDArray[np.float64]): Input array.

        Returns:
            npt.NDArray[np.float64]: Softplus of the input with numerical stabilization.
        """
        return np.logaddexp(x, 0)

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
    def median_trick(
        x: T_x,
        mode: str = "auto",
    ) -> float:
        """
        Compute the median trick.

        Args:
            x (npt.NDArray[np.float64]): Input array.
            mode (str, optional): Mode for computation. Defaults to "auto".
                - "auto": Automatically detect the type of x and compute accordingly.
                - "numpy": Use numpy for computation.
                - "torch": Use torch for computation.

        Returns:
            float: Median trick.
        """
        match mode:
            case "auto":
                if isinstance(x, torch.Tensor):
                    return MedianTrick.median_trick_torch(x)
                elif isinstance(x, np.ndarray):
                    return MedianTrick.median_trick_numpy(x)
                elif isinstance(x, list):
                    return MedianTrick.median_trick_numpy(np.array(x))
                else:
                    raise TypeError(
                        "Input must be a numpy array, torch tensor, or list."
                    )
            case "numpy":
                return MedianTrick.median_trick_numpy(x)
            case "torch":
                return MedianTrick.median_trick_torch(x)
            case _:
                raise ValueError("mode should be 'numpy', 'torch', or 'auto'.")

    @staticmethod
    def calculate_mmd(
        x: T_x,
        y: T_y,
        sigma: float = 1.0,
        /,
        mode: str = "auto",
        *,
        batch_size: int = 1_000,
    ) -> float:
        match mode:
            case "auto":
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=torch.float64)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float64)
                return BatchedCalculateMMDTorch.calculate(x, y, sigma, batch_size)
            case "numpy":
                return CalculateMMDNumpy.calculate(x, y, sigma)
            case "torch":
                return CalculateMMDTorch.calculate(x, y, sigma)
            case "batch_torch":
                return BatchedCalculateMMDTorch.calculate(x, y, sigma, batch_size)
            case "batch_numpy":
                return BatchedCalculateMMDTorch.calculate(x, y, sigma, batch_size)
            case _:
                raise ValueError("mode should be 'numpy', 'torch', or 'auto'.")

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

    @cache
    @staticmethod
    def gold_standard(
        model_name: str,
        posteriordb_path: str,
    ) -> npt.NDArray[np.float64]:
        return PosteriorDBToolbox(posteriordb_path).get_gold_standard(model_name)

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
            Callable[[Any], Any]:
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

    @staticmethod
    def convert_txt_res_to_markdown(model_name: str, mcmc_env: str) -> None:
        """
        Convert the text result to markdown format.

        Args:
            model_name (str): Model name.
            mcmc_env (str): MCMC environment.
                - mala
                - mala_esjd
                - barker
                - barker_esjd
        """
        mcmc_env_markdown_title = {
            "mala": "MALA",
            "mala_esjd": "MALA ESJD",
            "barker": "Barker",
            "barker_esjd": "Barker ESJD",
        }

        if mcmc_env not in mcmc_env_markdown_title.keys():
            raise ValueError(
                f"mcmc_env should be one of {list(mcmc_env_markdown_title.keys())}"
            )

        with open(f"{model_name}-{model_name}_{mcmc_env}_mmd.txt") as f:
            lines = f.readlines()

        markdown_exp = ["| env | random seed| mmd |", "| :---: | :---: | :---: |"]
        markdown_total = ["| env | mean | se |", "| :---: | :---: | :---: |"]

        for i in lines:
            if re.match(r"^\d", i):
                markdown_exp.append(
                    f"| ddpg_{mcmc_env} | " + lines[0].replace(", ", " | ").strip("\n")
                )

        markdown_total.append(
            f"| ddpg_{mcmc_env} | "
            + f"{float(lines[-2].strip("\n").split(": ")[-1]):.5f}"
            + " | "
            + f"{float(lines[-1].strip("\n").split(": ")[-1]):.5f}"
            + " |"
        )

        with open(f"{model_name}-{model_name}_{mcmc_env}_mmd.md", "w") as f:
            f.write(f"#### {mcmc_env_markdown_title[mcmc_env]}\n\n")

            for i in markdown_exp:
                f.write(i + "\n")

            f.write("\n")

            for i in markdown_total:
                f.write(i + "\n")

    @staticmethod
    def bold_markdown(
        input_file_path: str,
        output_file_path: str = "output.md",
        model_key_name: str = "Model",
        flex_key_name: str = "mala Median",
        baseline_key_name: str = "baseline Median",
    ) -> None:
        """
        Convert a markdown table to a new markdown table with bold values for the flex_key_name and baseline_key_name.

        Args:
            input_file_path (str): Path to the input markdown file.
            output_file_path (str): Path to the output markdown file.
            model_key_name (str): Name of the model key in the table.
            flex_key_name (str): Name of the flexible key in the table.
            baseline_key_name (str): Name of the baseline key in the table.
        """
        # Read markdown file content
        with open(input_file_path, "r") as f:
            md_table = f.read()

        # Remove the separator line in the header (lines like ---)
        lines = [
            line
            for line in md_table.strip().split("\n")
            if not re.match(r"^\s*\|?\s*-+", line)
        ]

        # Read a table using pandas
        df = pd.read_csv(StringIO("\n".join(lines)), sep="|", skipinitialspace=True)

        # Remove extra empty columns on both sides
        df.columns = df.columns.str.strip()
        if "" in df.columns:
            df = df.drop(columns=[""])
        # Delete columns containing Unnamed
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # Remove extra rows (Model column is the alignment line)
        df = df[~df[model_key_name].str.contains(":-", na=False)].reset_index(drop=True)

        # Check column names
        logger.info(f"Columns: {df.columns.tolist()}")

        # Bold logic
        def bold_compare(row):
            try:
                mala = float(row[flex_key_name])
                base = float(row[baseline_key_name])
            except ValueError:
                return pd.Series(
                    {
                        flex_key_name: row[flex_key_name],
                        baseline_key_name: row[baseline_key_name],
                    }
                )

            if mala < base:
                mala_fmt = f"**{mala}**"
                base_fmt = f"{base}"
            elif mala > base:
                mala_fmt = f"{mala}"
                base_fmt = f"**{base}**"
            else:
                mala_fmt = f"{mala}"
                base_fmt = f"{base}"

            return pd.Series({flex_key_name: mala_fmt, baseline_key_name: base_fmt})

        # Apply bold logic
        df[[flex_key_name, baseline_key_name]] = df.apply(bold_compare, axis=1)

        # Export markdown
        with open(output_file_path, "w") as f:
            f.write(df.to_markdown(index=False))
