import os
import shutil
from itertools import product
from typing import Dict

import jinja2
import numpy as np

from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox


def create_model_result_dir(model_name: str) -> str:
    """
    Create a directory for the model result if it doesn't exist.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The path to the model result directory.
    """
    model_result_dir = f"./whole_results/{model_name}"
    if not os.path.exists("./whole_results"):
        os.makedirs("./whole_results")
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    if not os.path.exists(f"{model_result_dir}/config"):
        os.makedirs(f"{model_result_dir}/config")

    return model_result_dir


def output_initial_step_size(model_sample_dim: int) -> float:
    """
    Output the initial step size for the model.

    Args:
        model_sample_dim (int): The dimension of the model sample.

    Returns:
        float: The initial step size.
    """
    if model_sample_dim <= 10:
        return 0.5
    elif model_sample_dim >= 100:
        return 0.01
    else:
        # Log-linear interpolate between 0.5 (dim=10) and 0.01 (dim=100)
        log_eps = np.interp(
            np.log(model_sample_dim),
            [np.log(10), np.log(100)],
            [np.log(0.5), np.log(0.01)],
        )
        return float(np.exp(log_eps))


def generate_files(
    model_name: str,
    repeat_num: int,
    template_root_dir: str = "./template",
    posteriordb_path: str = "../../posteriordb/posterior_database",
) -> None:
    """
    Generate the config files for the model.

    Args:
        model_name (str): The name of the model.
        repeat_num (int): The number of times to repeat the experiment.
        template_root_dir (str): The root directory for the template files.
        posteriordb_path (str): The path to the posterior database.
    """
    model_result_dir = create_model_result_dir(model_name)

    rl_algorithm_list = ["ddpg"]
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
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

    for random_seed in range(repeat_num):
        for rl_algorithm, mcmc_env in product(rl_algorithm_list, mcmc_env_list):
            hyperparameter_context: Dict[str, str] = {
                "exp_name": exp_name_dict[mcmc_env],
                "random_seed": str(random_seed),
                "env_id": mcmc_env_dict[mcmc_env],
                "actor_learning_rate": "1e-5",
            }
            hyperparameter_template_path = (
                f"{template_root_dir}/config_template_pdb.toml"
            )

            with open(hyperparameter_template_path, "r") as file:
                hyperparameter_template = jinja2.Template(file.read())

            config_content = hyperparameter_template.render(hyperparameter_context)

            hyperparameter_config_path = f"{model_result_dir}/config/{rl_algorithm}_{mcmc_env}/{rl_algorithm}_{mcmc_env}_seed_{random_seed}.toml"
            Toolbox.create_folder(hyperparameter_config_path)
            with open(
                hyperparameter_config_path,
                "w",
            ) as file:
                file.write(config_content)

            shutil.copy(f"{template_root_dir}/actor.toml", f"{model_result_dir}/config")
            shutil.copy(
                f"{template_root_dir}/critic.toml", f"{model_result_dir}/config"
            )

            pdb_run_path = f"{template_root_dir}/template_pdb_run.py"
            with open(pdb_run_path, "r") as file:
                pdb_run_template = jinja2.Template(file.read())
            pdb_run_context = {
                "model_name": model_name,
                "posteriordb_path": posteriordb_path,
                "rl_algorithm": rl_algorithm,
                "mcmc_env": mcmc_env,
                "random_seed": random_seed,
                "hyperparameter_config_path": f"./config/{rl_algorithm}_{mcmc_env}/{rl_algorithm}_{mcmc_env}_seed_{random_seed}.toml",
                "actor_config_path": "./config/actor.toml",
                "critic_config_path": "./config/critic.toml",
            }
            pdb_run_content = pdb_run_template.render(pdb_run_context)
            pdb_run_path = f"{model_result_dir}/run_pdb_{rl_algorithm}_{mcmc_env}_seed_{random_seed}.py"
            with open(pdb_run_path, "w") as file:
                file.write(pdb_run_content)

            bash_template_path = f"{template_root_dir}/template.run-pdb.sh"
            bash_context = {
                "model_name": model_name,
                "rl_algorithm": rl_algorithm,
                "mcmc_env": mcmc_env,
                "random_seed": random_seed,
            }
            with open(bash_template_path, "r") as file:
                bash_template = jinja2.Template(file.read())
            bash_content = bash_template.render(bash_context)
            bash_script_path = f"{model_result_dir}/run_bash_{rl_algorithm}_{mcmc_env}_seed_{random_seed}.sh"
            with open(bash_script_path, "w") as file:
                file.write(bash_content)


if __name__ == "__main__":
    posteriordb_path = "./posteriordb/posterior_database"

    pdb_toolbox = PosteriorDBToolbox(posteriordb_path)
    model_name_list = pdb_toolbox.get_model_name_with_gold_standard()

    for model_name in model_name_list:
        posteriordb_path = "../../posteriordb/posterior_database"
        repeat_num = 10
        template_root_dir = "./template"
        generate_files(
            model_name=model_name,
            repeat_num=repeat_num,
            template_root_dir=template_root_dir,
            posteriordb_path=posteriordb_path,
        )
