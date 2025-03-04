import argparse
import random
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeAliasType

import numpy as np
import pandas as pd
import torch
from numpy import typing as npt

from ..envs import BarkerEnv, BarkerESJDEnv, MALAEnv, MALAESJDEnv
from ..learning.preparation import PosteriorDBFunctionsGenerator
from .utils import CalculateMMD, Toolbox

EnvType = TypeAliasType(
    "EnvType", Type[BarkerEnv] | Type[BarkerESJDEnv] | Type[MALAEnv] | Type[MALAESJDEnv]
)
EnvInstanceType = TypeAliasType(
    "EnvInstanceType", BarkerEnv | BarkerESJDEnv | MALAEnv | MALAESJDEnv
)


class BenchmarkBase(ABC):
    _mcmc_envs: Dict[str, EnvType] = {
        "mala": MALAEnv,
        "barker": BarkerEnv,
        "mala_esjd": MALAESJDEnv,
        "barker_esjd": BarkerESJDEnv,
    }

    def __init__(self) -> None:
        self.args = self.make_parser().parse_args()

        self.env_name = self.args.mcmc_env
        self.model_name = self.args.model_name
        self.posteriordb_path = self.args.posteriordb_path
        self.random_seed = self.args.random_seed
        self.step_size = Toolbox.inverse_softplus(np.array([self.args.step_size]))

        self.env = self.make_env()

    def fixed_seed(self) -> None:
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def make_target_pdf(self) -> Tuple[
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ]:
        posteriordb_generator = PosteriorDBFunctionsGenerator(
            model_name=self.model_name,
            posteriordb_path=self.posteriordb_path,
            posterior_data=None,
        )
        log_target_pdf = posteriordb_generator.make_log_pdf()
        grad_log_target_pdf = posteriordb_generator.make_grad_log_pdf()

        return log_target_pdf, grad_log_target_pdf

    def make_env(self) -> EnvInstanceType:
        self.fixed_seed()

        log_target_pdf, grad_log_target_pdf = self.make_target_pdf()

        initial_sample = 0.1 * np.ones(2)
        config = {
            "log_target_pdf_unsafe": log_target_pdf,
            "grad_log_target_pdf_unsafe": grad_log_target_pdf,
            "initial_sample": initial_sample,
            "initial_covariance": None,
            "initial_step_size": self.step_size,
            "total_timesteps": 500_000,
            "max_steps_per_episode": 500,
            "log_mode": True,
        }

        mcmc = self._mcmc_envs[self.env_name](**config)
        mcmc.reset(seed=self.random_seed)

        return mcmc

    def run_mcmc(self) -> None:
        for _ in range(self.env.total_timesteps):
            self.env.step(np.repeat(self.step_size, 2))

    def make_parser(self) -> argparse.ArgumentParser:
        """
        Create an argument parser for the benchmark experiment.

        Returns:
            argparse.ArgumentParser: The argument parser for the benchmark experiment
        """
        parser = argparse.ArgumentParser(
            description="Run MCMC experiment with different parameters"
        )
        parser.add_argument(
            "--random_seed",
            type=int,
            required=True,
            help="Random seed for the experiment",
        )
        parser.add_argument(
            "--step_size", type=float, default=10.0, help="Step size for MCMC"
        )
        parser.add_argument(
            "--model_name",
            type=str,
            default="test-laplace_1-test-laplace_1",
            help="Model name",
        )
        parser.add_argument(
            "--posteriordb_path",
            type=str,
            default="../../posteriordb/posterior_database",
            help="Path to posterior database",
        )
        parser.add_argument(
            "--mcmc_env", type=str, default="mala", help="MCMC environment to use"
        )

        return parser

    def get_gold_standard(self) -> npt.NDArray[np.float64]:
        return Toolbox.gold_standard(self.model_name, self.posteriordb_path)

    def write_results(
        self, output_path: str, res: float, *args: Any, **kwargs: Any
    ) -> None:
        """
        Write the results to a CSV file.

        Args:
            output_path (str): The output path for the CSV file
            res (float): The result value
        """
        file_path = Path(output_path)
        output_path_with_extension = file_path.with_name(
            f"{file_path.stem}_{self.model_name}_{self.env_name}_{self.args.step_size}_{self.random_seed}{file_path.suffix}"
        )

        with open(output_path_with_extension, "w") as f:
            f.write("random_seed,step_size,model_name,mcmc_env,res\n")
            f.write(
                f"{self.random_seed},{self.args.step_size},{self.model_name},{self.env_name},{res}\n"
            )

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> None:
        """
        Execute the benchmark experiment.

        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError("Method execute is not implemented")


class MMDBenchMark(BenchmarkBase):
    def execute(self) -> None:
        """
        Execute the benchmark experiment.
        """
        self.run_mcmc()

        gs = self.get_gold_standard()
        mmd = CalculateMMD.calculate(gs, self.env.store_accepted_sample[-len(gs) :])
        self.write_results("mmd.csv", mmd)


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
        """
        file_path = Path(output_path)
        output_path_with_extension = file_path.with_name(
            f"{file_path.stem}_{self.model_name}_{self.env_name}_{self.args.step_size}_{self.random_seed}{file_path.suffix}"
        )

        with open(output_path_with_extension, "w") as f:
            f.write("random_seed,step_size,model_name,mcmc_env,res,esjd\n")
            f.write(
                f"{self.random_seed},{self.args.step_size},{self.model_name},{self.env_name},{res},{esjd}\n"
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
    @staticmethod
    def generate_parallel_jobs(
        model_dict: Dict[str, str],
        mcmc_envs: List[str],
        step_sizes: List[float],
        random_seeds: List[int],
        posteriordb_path: str,
        jobs_path: str,
        script_path: str,
    ) -> None:
        """
        Generate parallel jobs for the benchmark. The parallel jobs are generated for the benchmark.

        Args:
            model_dict (Dict[str, str]): The model dictionary
            mcmc_envs (List[str]): The MCMC environments
            step_sizes (List[float]): The step sizes
            random_seeds (List[int]): The random seeds
            posteriordb_path (str): The path to the posterior database
            jobs_path (str): The path to the jobs file
            script_path (str): The path to the script file
        """
        with open(jobs_path, "w") as f:
            for model_name in model_dict:
                for mcmc_env in mcmc_envs:
                    for step_size in step_sizes:
                        for random_seed in random_seeds:
                            f.write(
                                f"python {Path(script_path).name} --random_seed {random_seed} --step_size {step_size} --model_name {model_dict[model_name]} --posteriordb_path {posteriordb_path} --mcmc_env {mcmc_env}\n"
                            )

    @staticmethod
    def write_script(script_path: str, benchmark_type: str) -> None:
        """
        Write the MMD script to calculate the MMD for the benchmark.
        The MMD script is written to calculate the MMD for the benchmark.
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
    def optimal_step_size(data_csv_file_path: str = "merged_data.csv"):
        return (
            pd.read_csv(data_csv_file_path)
            .groupby(["mcmc_env", "step_size"])["res"]
            .mean()
            .reset_index()
            .sort_values(["mcmc_env", "res"], ascending=[True, False])
            .groupby("mcmc_env")
            .head(1)
        )
