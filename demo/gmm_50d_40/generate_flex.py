import math
import os
import shutil
from itertools import product
from typing import Dict

import jinja2
import numpy as np

from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox
from pyrlmala.utils.target import AutoStanTargetPDF


def generate_files(
    model_name: str,
    repeat_num: int,
    template_root_dir: str = "./template",
) -> None:
    """
    Generate the config files for the model.

    Args:
        model_name (str): The name of the model.
        repeat_num (int): The number of times to repeat the experiment.
        template_root_dir (str): The root directory for the template files.
        posteriordb_path (str): The path to the posterior database.
    """

    rl_algorithm_list = ["ddpg"]
    mcmc_env_list = ["mala", "mala_esjd"]
    exp_name_dict = {
        "mala": "RLMALA",
        "mala_esjd": "RLMALAESJD",
        "barker": "RLBarker",
        "barker_esjd": "RLBarkerESJD",
    }
    mcmc_env_dict = {
        "mala": "MALAEnv-v1.0",
        "mala_esjd": "MALAESJDEnv-v1.0",
        "barker": "BarkerEnv-v1.0",
        "barker_esjd": "BarkerESJDEnv-v1.0",
    }

    for rl_algorithm, mcmc_env in product(rl_algorithm_list, mcmc_env_list):
        for random_seed in range(repeat_num):
            hyperparameter_context: Dict[str, str] = {
                "exp_name": exp_name_dict[mcmc_env],
                "random_seed": str(random_seed),
                "env_id": mcmc_env_dict[mcmc_env],
                "actor_learning_rate": "1e-6",
            }
            hyperparameter_template_path = (
                "./config/config_template.toml"
            )

            with open(hyperparameter_template_path, "r") as file:
                hyperparameter_template = jinja2.Template(file.read())

            config_content = hyperparameter_template.render(hyperparameter_context)

            hyperparameter_config_path = f"./config/{rl_algorithm}_{mcmc_env}/{rl_algorithm}_{mcmc_env}_seed_{random_seed}.toml"
            Toolbox.create_folder(hyperparameter_config_path)
            with open(
                hyperparameter_config_path,
                "w",
            ) as file:
                file.write(config_content)

            jax_run_path = f"{template_root_dir}/template_flex.py"
            with open(jax_run_path, "r") as file:
                jax_run_template = jinja2.Template(file.read())
            jax_run_context = {
                "random_seed": random_seed,
                "mcmc_env": mcmc_env,
            }
            jax_run_content = jax_run_template.render(jax_run_context)
            jax_run_path = f"./run_jax_{rl_algorithm}_{mcmc_env}_{random_seed}.py"
            with open(jax_run_path, "w") as file:
                file.write(jax_run_content)

            bash_template_path = f"{template_root_dir}/template.run-batch.sh"
            bash_context = {
                "model_name": model_name,
                "rl_algorithm": rl_algorithm,
                "mcmc_env": mcmc_env,
                "random_seed": random_seed,
            }
            with open(bash_template_path, "r") as file:
                bash_template = jinja2.Template(file.read())
            bash_content = bash_template.render(bash_context)
            bash_script_path = f"./run_bash_{rl_algorithm}_{mcmc_env}_{random_seed}.sh"
            with open(bash_script_path, "w") as file:
                file.write(bash_content)


if __name__ == "__main__":
    model_name_list = "gmm_50d_40"

    for model_name in model_name_list:
        repeat_num = 10
        template_root_dir = "./template"
        generate_files(
            model_name=model_name,
            repeat_num=repeat_num,
            template_root_dir=template_root_dir,
        )
