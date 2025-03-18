import itertools
import os
from typing import Dict, Iterator

import jinja2
import numpy as np
from numpy import typing as npt
from tqdm.auto import tqdm

from ..learning import LearningFactory
from .utils import CalculateMMD, Toolbox


class FlexibleConfigGenerator:
    """
    A class to generate
    configuration files based on a template and a context dictionary.
    """

    @staticmethod
    def generate_config(
        template_path: str, output_path: str, context: Dict[str, str | int]
    ) -> None:
        """
        Generates a configuration file based on a template and a context dictionary.
        The template is rendered with the context, and the resulting content is
        written to the output file.
        The context dictionary should contain the following keys:
            - exp_name: The name of the experiment.
            - random_seed: The random seed for the experiment.
            - env_id: The environment ID for the experiment.

        Args:
            template_path (str): The path to the template file.
            output_path (str): The path to the output file.
            context (Dict[str, str  |  int]): The context dictionary to render the template.
        """
        with open(template_path, "r") as file:
            template = jinja2.Template(file.read())

        config_content = template.render(context)

        with open(output_path, "w") as file:
            file.write(config_content)

    @staticmethod
    def generate_context(
        output_root_path: str, repeat_count: int
    ) -> Iterator[Dict[str, str | int]]:
        """
        Generates a context dictionary for the configuration files.
        The context dictionary contains the following keys:
            - exp_name: The name of the experiment.
            - random_seed: The random seed for the experiment.
            - env_id: The environment ID for the experiment.
        The output path is generated based on the output_root_path and the experiment name.

        Args:
            output_root_path (str): The root path for the output files.
            repeat_count (int): The number of times to repeat the experiment.

        Returns:
            Iterator[Dict[str, str | int]]: An iterator that yields context dictionaries.
        """

        mala_config_generator = (
            {
                "context": {
                    "exp_name": "RLMALA",
                    "random_seed": i,
                    "env_id": "MALAEnv-v1.0",
                },
                "output_path": f"{output_root_path}/ddpg_mala/ddpg_mala_seed_{i}.toml",
            }
            for i in range(repeat_count)
        )
        mala_esjd_config_generator = (
            {
                "context": {
                    "exp_name": "RLMALAESJD",
                    "random_seed": i,
                    "env_id": "MALAESJDEnv-v1.0",
                },
                "output_path": f"{output_root_path}/ddpg_mala_esjd/ddpg_mala_esjd_seed_{i}.toml",
            }
            for i in range(repeat_count)
        )
        barker_config_generator = (
            {
                "context": {
                    "exp_name": "RLBarker",
                    "random_seed": i,
                    "env_id": "BarkerEnv-v1.0",
                },
                "output_path": f"{output_root_path}/ddpg_barker/ddpg_barker_seed_{i}.toml",
            }
            for i in range(repeat_count)
        )
        barker_esjd_config_generator = (
            {
                "context": {
                    "exp_name": "RLBarkerESJD",
                    "random_seed": i,
                    "env_id": "BarkerESJDEnv-v1.0",
                },
                "output_path": f"{output_root_path}/ddpg_barker_esjd/ddpg_barker_esjd_seed_{i}.toml",
            }
            for i in range(repeat_count)
        )

        combined_config_generator = itertools.chain(
            mala_config_generator,
            mala_esjd_config_generator,
            barker_config_generator,
            barker_esjd_config_generator,
        )

        return combined_config_generator

    @staticmethod
    def generate(
        template_path: str, output_root_path: str, repeat_count: int = 5
    ) -> None:
        """
        Executes the configuration generation process.
        It creates the output directory if it doesn't exist and generates
        configuration files based on the template and context dictionary.
        The context dictionary is generated using the generate_context method.
        The output path is generated based on the output_root_path and the experiment name.
        The configuration files are generated for different experiments:
            - RLMALA
            - RLMALAESJD
            - RLBarker
            - RLBarkerESJD
        The output files are saved in the following directories:
            - ddpg_mala
            - ddpg_mala_esjd
            - ddpg_barker
            - ddpg_barker_esjd
        The output files are named based on the experiment name and the random seed.
        The output files are saved in the following format:
            - ddpg_mala_seed_{random_seed}.toml
            - ddpg_mala_esjd_seed_{random_seed}.toml
            - ddpg_barker_seed_{random_seed}.toml
            - ddpg_barker_esjd_seed_{random_seed}.toml


        Args:
            template_path (str): The path to the template file.
            output_root_path (str): The root path for the output files.
            repeat_count (int, optional): The number of times to repeat the experiment. Defaults to 5.
        """
        combined_config_generator = FlexibleConfigGenerator.generate_context(
            output_root_path, repeat_count
        )

        for config in combined_config_generator:
            Toolbox.create_folder(config["output_path"])
            FlexibleConfigGenerator.generate_config(
                template_path=template_path,
                output_path=config["output_path"],
                context=config["context"],
            )


class FlexibleBatchRunner:
    def __init__(
        self, model_name: str, posteriordb_path: str, load_policy: str = "swa"
    ) -> None:
        """
        A class to run a batch of experiments with different configurations.
        It uses the FlexibleConfigGenerator to generate configuration files
        and the LearningFactory to create learning instances.
        The class provides methods to calculate MMD (Maximum Mean Discrepancy)
        and write results to a file.
        The class also provides a method to run the batch of experiments
        with the specified configurations.

        Args:
            model_name (str): The name of the model to be used in the experiments.
            posteriordb_path (str): The path to the posterior database.
            load_policy (str, optional): The policy to be loaded. Defaults to "swa".
        """
        self.model_name = model_name
        self.posteriordb_path = posteriordb_path
        self.load_policy = load_policy

    def calculate_mmd(
        self,
        random_seed: int,
        gold_standard: npt.NDArray[np.float64],
        step_size: float,
        mcmc_env: str,
    ) -> float:
        """
        Calculates the Maximum Mean Discrepancy (MMD) between the predicted
        samples and the gold standard samples.

        Args:
            random_seed (int): The random seed for the experiment.
            gold_standard (npt.NDArray[np.float64]): The gold standard samples.
            step_size (float): The step size for the MCMC algorithm.
            mcmc_env (str): The MCMC environment to be used in the experiment.

        Returns:
            float: The calculated MMD value.
        """
        sample_dim = 2
        initial_sample = 0.1 * np.ones(sample_dim)
        initial_step_size = np.array([step_size])
        algorithm = "ddpg"

        learning_instance = LearningFactory.create_learning_instance(
            algorithm=algorithm,
            model_name=self.model_name,
            posteriordb_path=self.posteriordb_path,
            initial_sample=initial_sample,
            initial_step_size=initial_step_size,
            hyperparameter_config_path=f"./config/{algorithm}_{mcmc_env}/{algorithm}_{mcmc_env}_seed_{random_seed}.toml",
            actor_config_path="./config/actor.toml",
            critic_config_path="./config/critic.toml",
            verbose=False,
        )

        learning_instance.train()
        learning_instance.predict(self.load_policy)

        # Calculate MMD
        predicted_sample = learning_instance.predicted_observation[:, 0:sample_dim]
        mmd = CalculateMMD.calculate(gold_standard, predicted_sample)

        return mmd

    @staticmethod
    def write_results(random_seed: int, mmd: float, save_file_path: str) -> None:
        """
        Writes the results of the experiment to a file.

        Args:
            random_seed (int): The random seed for the experiment.
            mmd (float): The calculated MMD value.
            save_file_path (str): The path to the file where the results will be saved.
        """
        with open(save_file_path, "a+") as f:
            f.write(f"{random_seed}, {mmd}\n")

    def run(
        self,
        mcmc_env: str,
        step_size: float,
        repeat_count: int = 5,
        save_root_path: str = ".",
        template_path: str = "./config/template.toml",
        output_root_path: str = "./config",
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
        FlexibleConfigGenerator.generate(template_path, output_root_path, repeat_count)

        save_file_path = os.path.join(
            save_root_path, f"{self.model_name}_{mcmc_env}_mmd.txt"
        )
        Toolbox.create_folder(save_file_path)

        random_seeds = range(repeat_count)
        gold_standard = Toolbox.gold_standard(self.model_name, self.posteriordb_path)

        mmd_res = np.empty(len(random_seeds))

        for i in tqdm(random_seeds):
            try:
                mmd = self.calculate_mmd(i, gold_standard, step_size, mcmc_env)
                mmd_res[i] = mmd
                self.write_results(i, mmd, save_file_path)
            except Exception as e:
                print(f"Error in seed {i}: {e}")

        with open(save_file_path, "a+") as f:
            f.write(f"Mean: {mmd_res.mean()}\n")
            f.write(f"SE: {mmd_res.std(ddof=1) / np.sqrt(len(mmd_res))}\n")
