import argparse
import random
from pathlib import Path

import numpy as np
import torch

from ..envs import BarkerEnv, MALAEnv
from ..learning.preparation import PosteriorDBFunctionsGenerator
from .utils import CalculateMMD, Toolbox


class MCMCBenchmark:
    @staticmethod
    def const_sampler(
        random_seed: int,
        step_size: float,
        model_name: str,
        posteriordb_path: str,
        mcmc_env: str,
    ) -> float:
        """
        Run MCMC experiment with different parameters.

        Args:
            random_seed (int): Random seed for the experiment
            step_size (float): Step size for MCMC
            model_name (str): Model name
            posteriordb_path (str): Path to posterior database
            mcmc_env (str): MCMC environment to use

        Raises:
            ValueError: If the MCMC environment is invalid or not supported by the benchmark experiment

        Returns:
            float: The MMD value
        """
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        posteriordb_generator = PosteriorDBFunctionsGenerator(
            model_name=model_name,
            posteriordb_path=posteriordb_path,
            posterior_data=None,
        )
        log_target_pdf = posteriordb_generator.make_log_pdf()
        grad_log_target_pdf = posteriordb_generator.make_grad_log_pdf()

        initial_sample = 0.1 * np.ones(2)
        initial_step_size = np.array([step_size])

        config = {
            "log_target_pdf_unsafe": log_target_pdf,
            "grad_log_target_pdf_unsafe": grad_log_target_pdf,
            "initial_sample": initial_sample,
            "initial_covariance": None,
            "initial_step_size": Toolbox.inverse_softplus(initial_step_size),
            "total_timesteps": 500_000,
            "max_steps_per_episode": 500,
            "log_mode": True,
        }

        env_classes = {"mala": MALAEnv, "barker": BarkerEnv}
        if mcmc_env not in env_classes:
            raise ValueError(f"Invalid MCMC environment: {mcmc_env}")

        mcmc = env_classes[mcmc_env](**config)
        mcmc.reset(seed=random_seed)

        for _ in range(mcmc.total_timesteps):
            mcmc.step(np.repeat(Toolbox.inverse_softplus(initial_step_size), 2))
        gs = Toolbox.gold_standard(model_name, posteriordb_path)
        mmd = CalculateMMD.calculate(gs, mcmc.store_accepted_sample[-len(gs) :])

        return mmd

    @staticmethod
    def make_parser() -> argparse.ArgumentParser:
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

    @staticmethod
    def write_mmd(
        output_path: str,
        mmd: float,
        args: argparse.Namespace,
    ) -> None:
        """
        Write the MMD value to a CSV file.

        Args:
            output_path (str): The output path for the CSV file
            mmd (float): The MMD value
            args (argparse.Namespace): The arguments for the benchmark experiment
        """
        file_path = Path(output_path)
        output_path_with_extension = file_path.with_name(
            f"{file_path.stem}_{args.model_name}_{args.mcmc_env}_{args.step_size}_{args.random_seed}{file_path.suffix}"
        )

        with open(output_path_with_extension, "w") as f:
            f.write("random_seed,step_size,model_name,mcmc_env,mmd\n")
            f.write(
                f"{args.random_seed},{args.step_size},{args.model_name},{args.mcmc_env},{mmd}\n"
            )

    @staticmethod
    def execute() -> None:
        """
        Execute the benchmark experiment.
        The benchmark experiment is executed by parsing the arguments and writing the MMD value to a CSV file.
        """
        parser = MCMCBenchmark.make_parser()
        args = parser.parse_args()

        mmd = MCMCBenchmark.const_sampler(
            args.random_seed,
            args.step_size,
            args.model_name,
            args.posteriordb_path,
            args.mcmc_env,
        )

        MCMCBenchmark.write_mmd("mmd.csv", mmd, args)


class BenchmarkFactory:
    """
    Factory class to generate benchmark files for MCMC algorithms.

    Attributes:
        _model_list (List[str]): The list of model names.
        _mcmc_envs (Dict[str, str]): The dictionary of MCMC environments.
        _random_seed (int): The random seed.
        _step_size_list (List[float]): The list of step sizes.
    """

    _model_dict = {
        "gaussian": "test-multivariant_normal-test-multivariant_normal",
        "laplace_1": "test-laplace_1-test-laplace_1",
        "laplace_2": "test-laplace_2-test-laplace_2",
        "laplace_4": "test-laplace_4-test-laplace_4",
    }
    _model_dict = {"laplace_1": "test-laplace_1-test-laplace_1"}
    _mcmc_envs = {
        "mala": "MALAEnv",
        "barker": "BarkerEnv",
    }
    _mcmc_envs = {"barker": "BarkerEnv"}
    _random_seed = range(5)
    _step_size_list = [i if i != 0 else 0.1 for i in range(20)] + [0.5]

    _posteriordb_path = "../posteriordb/posterior_database"

    _jobs_path = "./jobs.txt"
    _mmd_script_path = "./calculate_mmd.py"

    @staticmethod
    def generate_parallel_jobs() -> None:
        """
        Generate parallel jobs for the benchmark.
        The parallel jobs are generated based on the model list, MCMC environments, random seeds, and step sizes.
        """
        with open(BenchmarkFactory._jobs_path, "w") as f:
            for model_name in BenchmarkFactory._model_dict:
                for mcmc_env in BenchmarkFactory._mcmc_envs:
                    for step_size in BenchmarkFactory._step_size_list:
                        for random_seed in BenchmarkFactory._random_seed:
                            f.write(
                                f"python {Path(BenchmarkFactory._mmd_script_path).name} --random_seed {random_seed} --step_size {step_size} --model_name {BenchmarkFactory._model_dict[model_name]} --posteriordb_path {BenchmarkFactory._posteriordb_path} --mcmc_env {mcmc_env}\n"
                            )

    @staticmethod
    def write_mmd_script() -> None:
        """
        Write the MMD script to calculate the MMD for the benchmark.
        The MMD script is written to calculate the MMD for the benchmark.
        """
        with open(BenchmarkFactory._mmd_script_path, "w") as f:
            f.write(
                "from pyrlmala.utils.benchmark import MCMCBenchmark\n\n\nMCMCBenchmark.execute()\n"
            )

    @staticmethod
    def execute() -> None:
        """
        Execute the benchmark.
        The benchmark is executed by writing the MMD script and generating parallel jobs.
        """
        BenchmarkFactory.write_mmd_script()
        BenchmarkFactory.generate_parallel_jobs()
