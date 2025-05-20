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


def get_magnitude(x: float) -> float:
    """
    Get the magnitude of a number.

    Args:
        x (float): The input number.

    Returns:
        float: The magnitude of the number.
    """
    if x == 0:
        return 0

    exponent = math.floor(math.log10(abs(x)))

    return 10**exponent


def create_model_result_dir(model_name: str) -> str:
    """
    Create a directory for the model result if it doesn't exist.

    Args:
        model_name (str): The name of the model.

    Returns:
        str: The path to the model result directory.
    """
    model_result_dir = f"./sensitivity/{model_name}"
    if not os.path.exists("./sensitivity"):
        os.makedirs("./sensitivity")
    if not os.path.exists(model_result_dir):
        os.makedirs(model_result_dir)
    if not os.path.exists(f"{model_result_dir}/config"):
        os.makedirs(f"{model_result_dir}/config")

    return model_result_dir


def output_initial_step_size(
    model_name: str,
    posteriordb_path: str = "./posteriordb/posterior_database",
    l: float = 1.65,
) -> float:
    target = AutoStanTargetPDF(model_name, posteriordb_path)
    sigma_inv = -target.hess_log_target_pdf(
        Toolbox.gold_standard(model_name, posteriordb_path).mean(axis=0)
    )
    d = sigma_inv.shape[0]

    lam_max = np.linalg.eigvalsh(sigma_inv).max()
    eps0 = l / np.sqrt(lam_max * d ** (1 / 3))
    initial_step_size = 29.0168 * eps0**3 - 25.6180 * eps0**2 + 3.0239 * eps0 + 1.3476

    return initial_step_size.item()


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

    for rl_algorithm, mcmc_env in product(rl_algorithm_list, mcmc_env_list):
        for random_seed in range(repeat_num):
            hyperparameter_context: Dict[str, str] = {
                "exp_name": exp_name_dict[mcmc_env],
                "random_seed": str(random_seed),
                "env_id": mcmc_env_dict[mcmc_env],
                "actor_learning_rate": "1e-6",
                "exploration_noise": str(
                    get_magnitude(output_initial_step_size(model_name))
                ),
            }
            hyperparameter_template_path = (
                f"{template_root_dir}/config_template_sensitivity.toml"
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
        shutil.copy(f"{template_root_dir}/critic.toml", f"{model_result_dir}/config")

        sensitivity_run_path = f"{template_root_dir}/template_sensitivity_run.py"
        with open(sensitivity_run_path, "r") as file:
            sensitivity_run_template = jinja2.Template(file.read())
        sensitivity_run_context = {
            "model_name": model_name,
            "posteriordb_path": posteriordb_path,
            "rl_algorithm": rl_algorithm,
            "mcmc_env": mcmc_env,
            "repeat_num": repeat_num,
        }
        sensitivity_run_content = sensitivity_run_template.render(
            sensitivity_run_context
        )
        sensitivity_run_path = (
            f"{model_result_dir}/run_sensitivity_{rl_algorithm}_{mcmc_env}.py"
        )
        with open(sensitivity_run_path, "w") as file:
            file.write(sensitivity_run_content)

        bash_template_path = f"{template_root_dir}/template.run-sensitivity.sh"
        bash_context = {
            "model_name": model_name,
            "rl_algorithm": rl_algorithm,
            "mcmc_env": mcmc_env,
        }
        with open(bash_template_path, "r") as file:
            bash_template = jinja2.Template(file.read())
        bash_content = bash_template.render(bash_context)
        bash_script_path = f"{model_result_dir}/run_bash_{rl_algorithm}_{mcmc_env}.sh"
        with open(bash_script_path, "w") as file:
            file.write(bash_content)


if __name__ == "__main__":
    posteriordb_path = "./posteriordb/posterior_database"

    sensitivity_toolbox = PosteriorDBToolbox(posteriordb_path)
    model_name_list = [
        i
        for i in sensitivity_toolbox.get_model_name_with_gold_standard()
        if "test" not in i
    ]

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
