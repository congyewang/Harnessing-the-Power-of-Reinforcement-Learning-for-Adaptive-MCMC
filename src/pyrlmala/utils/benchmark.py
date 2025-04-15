import os
import random
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import typing as npt
from tqdm.auto import trange

from ..envs import BarkerEnv, BarkerESJDEnv, MALAEnv, MALAESJDEnv
from .target import PosteriorDatabaseTargetPDF
from .types import EnvInstanceType, EnvType
from .utils import Toolbox


class BenchmarkBase(ABC):
    """
    Base class for benchmark experiments.
    """

    _mcmc_envs: Dict[str, EnvType] = {
        "mala": MALAEnv,
        "barker": BarkerEnv,
        "mala_esjd": MALAESJDEnv,
        "barker_esjd": BarkerESJDEnv,
    }

    def __init__(
        self,
        mcmc_env: str,
        model_name: str,
        posteriordb_path: str,
        random_seed: int,
        step_size: float,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the benchmark experiment.

        Args:
            mcmc_env (str): The name of the MCMC environment
                - "mala": MALA
                - "barker": Barker
                - "mala_esjd": MALA with ESJD
                - "barker_esjd": Barker with ESJD
            model_name (str): The name of the model
            posteriordb_path (str): The path to the posterior database
            random_seed (int): The random seed for reproducibility
            step_size (float): The step size for the MCMC algorithm
        """
        self.env_name = self.check_env(mcmc_env)
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.random_seed = random_seed
        self.step_size = step_size
        self.action = Toolbox.inverse_softplus(np.array([step_size]))
        self.verbose = verbose

        self.env = self.make_env()

    def check_env(self, env_name: str) -> str:
        """
        Check if the MCMC environment is valid.

        Args:
            env_name (str): The name of the MCMC environment

        Returns:
            str: The name of the MCMC environment

        Raises:
            ValueError: If the MCMC environment is not valid
        """
        if env_name not in self._mcmc_envs.keys():
            raise ValueError(
                f"Invalid MCMC environment '{env_name}'. Available options are: {list(self._mcmc_envs.keys())}"
            )
        else:
            return env_name

    def fixed_seed(self) -> None:
        """
        Set the random seed for reproducibility.
        """
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def make_target_pdf(self) -> Tuple[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]:
        """
        Create the target PDF and its gradient.

        Returns:
            Tuple[ Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], ]: The target PDF and its gradient
        """
        posteriordb_generator = PosteriorDatabaseTargetPDF(
            model_name=self.model_name,
            posteriordb_path=self.posteriordb_path,
        )

        return (
            posteriordb_generator.log_target_pdf,
            posteriordb_generator.grad_log_target_pdf,
        )

    def make_env(self) -> EnvInstanceType:
        """
        Create the MCMC environment.

        Returns:
            EnvInstanceType: The MCMC environment instance
        """
        self.fixed_seed()

        log_target_pdf, grad_log_target_pdf = self.make_target_pdf()

        initial_sample = 0.1 * np.ones(2)
        config: Dict[
            str,
            int
            | float
            | bool
            | npt.NDArray[np.float64]
            | Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
            | None,
        ] = {
            "log_target_pdf_unsafe": log_target_pdf,
            "grad_log_target_pdf_unsafe": grad_log_target_pdf,
            "initial_sample": initial_sample,
            "initial_covariance": None,
            "initial_step_size": self.action,
            "total_timesteps": 500_000,
            "max_steps_per_episode": 500,
            "log_mode": True,
        }

        mcmc = self._mcmc_envs[self.env_name](**config)
        mcmc.reset(seed=self.random_seed)

        return mcmc

    def run_mcmc(self) -> None:
        """
        Run the MCMC simulation for the specified number of timesteps.
        """
        for _ in trange(self.env.total_timesteps, disable=not self.verbose):
            self.env.step(np.repeat(self.action, 2))

    def get_gold_standard(self) -> npt.NDArray[np.float64]:
        """
        Get the gold standard samples from the posterior database.

        Returns:
            npt.NDArray[np.float64]: The gold standard samples
        """
        return Toolbox.gold_standard(self.model_name, self.posteriordb_path)

    @abstractmethod
    def output(self, *args: Any, **kwargs: Any) -> float:
        """
        Calculate the output of the benchmark experiment.

        Returns:
            float: The result of the benchmark experiment

        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError("Method execute is not implemented")


class MMDBenchMark(BenchmarkBase):
    def output(self) -> float:
        """
        Calculate the Maximum Mean Discrepancy (MMD) between the predicted samples and the gold standard samples.

        Returns:
            float: The calculated MMD value
        """
        self.run_mcmc()

        gs = self.get_gold_standard()
        mmd = Toolbox.calculate_mmd(gs, self.env.store_accepted_sample[-len(gs) :])

        return mmd


class MMDBatchRunner:
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str,
    ) -> None:
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path

    @staticmethod
    def write_results(
        model_name: str,
        env_name: str,
        random_seed: int,
        step_size: float,
        mmd: float,
        save_file_path: str,
    ) -> None:
        """
        Write the results of the experiment to a file.

        Args:
            model_name (str): The name of the model
            env_name (str): The name of the MCMC environment
            random_seed (int): The random seed for the experiment
            step_size (float): The step size for the MCMC algorithm
            mmd (float): The calculated MMD value
            save_file_path (str): The path to the file where the results will be saved
        """
        with open(save_file_path, "a+") as f:
            f.write(f"{random_seed},{step_size},{model_name},{env_name},{mmd}\n")

    @staticmethod
    def write_header(save_file_path: str) -> None:
        """
        Write the header of the CSV file.

        Args:
            save_file_path (str): The path to the file where the results will be saved
        """
        with open(save_file_path, "w") as f:
            f.write("random_seed,step_size,model_name,mcmc_env,mmd\n")

    def run(
        self,
        mcmc_env: str,
        step_size: float,
        repeat_count: int = 10,
        mode: str = "median",
        save_root_path: str = ".",
        verbose: bool = True,
    ) -> None:
        """
        Runs the batch of experiments with the specified configurations.

        Args:
            mcmc_env (str): The MCMC environment to be used in the experiment.
            step_size (float): The step size for the MCMC algorithm.
            repeat_count (int, optional): The number of times to repeat the experiment. Defaults to 5.
            save_root_path (str, optional): The root path for saving the results. Defaults to ".".
            template_path (str, optional): The path to the template file. Defaults to "./config/template.toml".
            output_root_path (str, optional): The root path for the output files. Defaults to "./config".
        """
        save_file_path = os.path.join(
            save_root_path, f"const_{self.model_name}_{mcmc_env}_{step_size}_mmd.csv"
        )
        Toolbox.create_folder(save_file_path)
        mmd_res = np.empty(repeat_count)

        self.write_header(save_file_path)

        for i in trange(repeat_count, disable=not verbose):
            try:
                mmd_benchmark = MMDBenchMark(
                    mcmc_env=mcmc_env,
                    model_name=self.model_name,
                    posteriordb_path=self.posteriordb_path,
                    random_seed=i,
                    step_size=step_size,
                    verbose=False,
                )
                mmd = mmd_benchmark.output()
                mmd_res[i] = mmd
                self.write_results(
                    self.model_name,
                    mcmc_env,
                    i,
                    step_size,
                    mmd,
                    save_file_path,
                )
            except Exception as e:
                print(f"Error in seed {i}: {e}")

        match mode:
            case "mean":
                metrics = np.mean(mmd_res)
                metrics_str = f"Mean: {metrics}\n"
            case "median":
                metrics = np.median(mmd_res)
                metrics_str = f"Median: {metrics}\n"
            case _:
                raise ValueError("Mode must be 'mean' or 'median'")

        with open(save_file_path, "a+") as f:
            f.write(metrics_str)
            f.write(f"SE: {mmd_res.std(ddof=1) / np.sqrt(len(mmd_res))}\n")


class RewardBenchmark(BenchmarkBase):
    def write_results(
        self,
        output_path: str,
        res: float,
        esjd: float,
    ) -> None:
        """
        Write the results to a CSV file.

        Args:
            output_path (str): The output path for the CSV file
            res (float): The result value
            esjd (float): The expected square jump distance
        """
        file_path = Path(output_path)
        output_path_with_extension = file_path.with_name(
            f"{file_path.stem}_{self.model_name}_{self.env_name}_{self.step_size}_{self.random_seed}{file_path.suffix}"
        )

        with open(output_path_with_extension, "w") as f:
            f.write("random_seed,step_size,model_name,mcmc_env,res,esjd\n")
            f.write(
                f"{self.random_seed},{self.step_size},{self.model_name},{self.env_name},{res},{esjd}\n"
            )

    def execute(self) -> None:
        """
        Execute the benchmark experiment.
        """
        self.run_mcmc()

        reward = np.mean(self.env.store_reward)
        esjd = Toolbox.expected_square_jump_distance(self.env.store_accepted_sample)

        self.write_results("reward_esjd.csv", reward.item(), esjd.item())


class BenchmarkGenerator:
    """
    Class to generate the benchmark script.
    """

    @staticmethod
    def write_script(script_path: str, benchmark_type: str) -> None:
        """
        Write the MMD script to calculate the MMD for the benchmark.
        The MMD script is written to calculate the MMD for the benchmark.

        Args:
            script_path (str): The path to the script file
            benchmark_type (str): The type of benchmark to run
        """
        _class_name = {
            "mmd": "MMDBenchMark",
            "reward": "RewardBenchmark",
        }

        if benchmark_type not in _class_name.keys():
            raise ValueError("Benchmark type must be one of 'mmd', 'reward'")

        with open(script_path, "w") as f:
            f.write(
                f"from pyrlmala.utils.benchmark import {_class_name[benchmark_type]}\n\n\n{_class_name[benchmark_type]}().execute()\n"
            )


class BenchmarkExporter:
    @staticmethod
    def merge_csv_files(data_csv_file_path: str = "merged_data.csv") -> None:
        """
        Merge the CSV files into a single CSV file

        Args:
            data_csv_file_path (str, optional): The path to the merged CSV file. Defaults to "merged_data.csv".
        """
        cmd = f"awk '(NR == 1) || (FNR > 1)' *.csv > {data_csv_file_path}"
        subprocess.run(cmd, shell=True, check=True)

    @staticmethod
    def group_data(
        data_csv_file_path: str = "merged_data.csv",
        output_file_path: str = "grouped_data.md",
    ) -> None:
        """
        Group the data and write it to a markdown file

        Args:
            data_csv_file_path (str, optional): The path to the merged CSV file. Defaults to "merged_data.csv".
            output_file_path (str, optional): The path to the output markdown file. Defaults to "grouped_data.md".
        """
        df = pd.read_csv(data_csv_file_path).sort_values(
            by=["mcmc_env", "step_size", "random_seed"], ascending=[True, True, True]
        )

        grouped_reward = (
            df.groupby(["mcmc_env", "step_size"])["res"]
            .agg(mean="mean", std="std", count="count")
            .assign(se=lambda df: df["std"] / np.sqrt(df["count"]))
            .drop(columns=["std", "count"])
            .rename(columns={"mean": "reward_mean", "se": "reward_se"})
        )
        grouped_esjd = (
            df.groupby(["mcmc_env", "step_size"])["esjd"]
            .agg(mean="mean", std="std", count="count")
            .assign(se=lambda df: df["std"] / np.sqrt(df["count"]))
            .drop(columns=["std", "count"])
            .rename(columns={"mean": "esjd_mean", "se": "esjd_se"})
        )

        grouped_df = grouped_reward.merge(grouped_esjd, on=["mcmc_env", "step_size"])
        grouped_df.to_markdown(
            output_file_path, tablefmt="github", index=True, floatfmt=".4f"
        )

    @staticmethod
    def optimal_step_size(data_csv_file_path: str = "merged_data.csv") -> pd.DataFrame:
        """
        Get the optimal step size for each MCMC environment.

        Args:
            data_csv_file_path (str, optional): The path to the merged CSV file. Defaults to "merged_data.csv".

        Returns:
            pd.DataFrame: The optimal step size for each MCMC environment
        """
        return (
            pd.read_csv(data_csv_file_path)
            .groupby(["mcmc_env", "step_size"])["res"]
            .mean()
            .reset_index()
            .sort_values(["mcmc_env", "res"], ascending=[True, False])
            .groupby("mcmc_env")
            .head(1)
        )


class BootstrapBenchmark:
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str,
        num: int = 10,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> None:
        """
        Initialize the BootstrapBenchmark class.

        Args:
            model_name (str): The name of the model
            posteriordb_path (str): The path to the posterior database
            num (int, optional): Number of bootstrap samples. Defaults to 10.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            verbose (bool, optional): Verbose output. Defaults to True.
        """
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.num = num
        self.random_seed = random_seed
        self.verbose = verbose

    def get_gold_standard(self) -> npt.NDArray[np.float64]:
        """
        Get the gold standard samples from the posterior database.

        Returns:
            npt.NDArray[np.float64]: The gold standard samples
        """
        return Toolbox.gold_standard(self.model_name, self.posteriordb_path)

    @staticmethod
    def bootstrap_sampling(
        gold_standard: npt.NDArray[np.floating],
        num: int = 10,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> npt.NDArray[np.floating]:
        """
        Bootstrap sampling for the MMD values.

        Args:
            gold_standard (npt.NDArray[np.floating]): Gold standard samples.
            num (int, optional): Number of bootstrap samples. Defaults to 10.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            verbose (bool, optional): Verbose output. Defaults to True.

        Returns:
            npt.NDArray[np.floating]:
        """
        mmd_values = np.empty(num)
        rng = np.random.default_rng(seed=random_seed)

        length_of_gold_standard = len(gold_standard)

        for i in trange(num, disable=not verbose):
            bootstrap_idx = rng.choice(
                length_of_gold_standard, length_of_gold_standard, replace=True
            )
            mmd_values[i] = Toolbox.calculate_mmd(
                gold_standard, gold_standard[bootstrap_idx]
            )

        return mmd_values

    @staticmethod
    def output_median_and_quantile(
        mmd_square_values: npt.NDArray[np.floating],
    ) -> Tuple[float, float, float]:
        """
        Calculate the median and quantiles of the MMD values.
        This function calculates the median and quantiles of the MMD values.

        Args:
            mmd_values (npt.NDArray[np.floating]): The MMD square values.

        Returns:
            Tuple[float, float, float]: Median and quantiles of the MMD values.
        """
        # mmd_values = np.sqrt(mmd_square_values)
        mmd_values = mmd_square_values

        mmd_median = np.median(mmd_values)
        mmd_quantile_25 = np.percentile(mmd_values, 25)
        mmd_quantile_75 = np.percentile(mmd_values, 75)

        return mmd_median.item(), mmd_quantile_25.item(), mmd_quantile_75.item()

    def write_results(
        self,
        output_path: str,
        mmd_median: float,
        mmd_left_quantile: float,
        mmd_right_quantile: float,
    ) -> None:
        """
        Write the results to a CSV file.

        Args:
            output_path (str): The output path for the CSV file
            mmd_median (float): The median of MMD values
            mmd_left_quantile (float): The left quantile of MMD values
            mmd_right_quantile (float): The right quantile of MMD values
        """
        file_path = Path(output_path)
        output_path_with_extension = file_path.with_name(
            f"{file_path.stem}_{self.model_name}_{self.random_seed}{file_path.suffix}"
        )

        with open(output_path_with_extension, "w") as f:
            f.write("model_name,mmd_median,mmd_25,mmd_75\n")
            f.write(f"{self.model_name},{mmd_median},{mmd_left_quantile},{mmd_right_quantile}\n")

    def execute(self) -> None:
        """
        Execute the benchmark experiment.
        """
        gs = self.get_gold_standard()
        mmd_values = self.bootstrap_sampling(
            gs, num=self.num, random_seed=self.random_seed, verbose=self.verbose
        )
        mmd_median, mmd_left_quantile, mmd_right_quantile = self.output_median_and_quantile(mmd_values)
        self.write_results("bootstrap_mmd.csv", mmd_median, mmd_left_quantile, mmd_right_quantile)
