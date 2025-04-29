import os
import shutil
from itertools import product
from typing import Dict

import jinja2

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


def output_initial_step_size(model_name: str) -> float:
    """
    Output a smooth initial step size based on model dimension using log-linear interpolation.

    Args:
        model_name (str): The name of the model.

    Returns:
        float: The initial step size for the model.
    """
    step_size_dict = {
        "arK-arK": 1.444885116153801,
        "mesquite-logmesquite_logvash": 1.3026946593080484,
        "earnings-logearn_interaction_z": 1.6773751049582213,
        "sblri-blr": 1.4871071073090385,
        "garch-garch11": 0.9281432387660594,
        "diamonds-diamonds": 0.9227758660447334,
        "nes1996-nes": 1.2985494241661342,
        "kidiq-kidscore_momhsiq": 1.8174460275661013,
        "low_dim_gauss_mix-low_dim_gauss_mix": 1.6616931193623956,
        "kidiq-kidscore_momiq": 2.052443006288145,
        "eight_schools-eight_schools_noncentered": 0.032297853500002846,
        "mcycle_gp-accel_gp": 0.00599514936951625,
        "hmm_example-hmm_example": 1.6537297541905724,
        "earnings-logearn_logheight_male": 1.837545169809095,
        "nes1980-nes": 1.265225407271176,
        "kidiq-kidscore_momhs": 2.0759726980979143,
        "arma-arma11": 1.7406216402946875,
        "earnings-logearn_interaction": 1.651080354596432,
        "eight_schools-eight_schools_centered": 0.11047533827053524,
        "hudson_lynx_hare-lotka_volterra": 1.0764082604774006,
        "kidiq_with_mom_work-kidscore_mom_work": 1.644490013581835,
        "gp_pois_regr-gp_regr": 2.025951555179607,
        "earnings-log10earn_height": 2.0627303833602317,
        "kidiq_with_mom_work-kidscore_interaction_z": 1.6633501545611526,
        "nes1984-nes": 1.2669750817391519,
        "nes1988-nes": 1.3105343227352024,
        "nes1972-nes": 1.2733228411882598,
        "mesquite-logmesquite": 1.2337267837763584,
        "nes1976-nes": 1.2812554197886838,
        "sblrc-blr": 1.4868078422514857,
        "mesquite-logmesquite_logvas": 1.2282993053337004,
        "kidiq_with_mom_work-kidscore_interaction_c": 1.641862772269111,
        "mesquite-logmesquite_logvolume": 1.872042337549332,
        "mesquite-logmesquite_logva": 1.4817593533186844,
        "earnings-logearn_height": 2.0349023121888217,
        "kidiq_with_mom_work-kidscore_interaction_c2": 1.6219512406858225,
        "kidiq-kidscore_interaction": 1.667698269421378,
        "nes2000-nes": 1.2835527435404193,
        "bball_drive_event_1-hmm_drive_1": 1.39602670527713,
        "mesquite-mesquite": 1.2386738530403414,
        "one_comp_mm_elim_abs-one_comp_mm_elim_abs": 0.7417336474220781,
        "nes1992-nes": 1.2956799224394653,
        "earnings-logearn_height_male": 1.8523044897309613,
        "bball_drive_event_0-hmm_drive_0": 1.2723081527243076,
        "earnings-earn_height": 1.7e-5,
        "kilpisjarvi_mod-kilpisjarvi": 3.75e-2,
        "gp_pois_regr-gp_pois_regr": 5.64e-1,
    }

    step_size = step_size_dict.get(model_name, 1e-2)

    return step_size


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
                "actor_learning_rate": "1e-5",
                "exploration_noise": str(2.0 * output_initial_step_size(model_name)),
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
        shutil.copy(f"{template_root_dir}/critic.toml", f"{model_result_dir}/config")

        pdb_run_path = f"{template_root_dir}/template_pdb_run.py"
        with open(pdb_run_path, "r") as file:
            pdb_run_template = jinja2.Template(file.read())
        pdb_run_context = {
            "model_name": model_name,
            "posteriordb_path": posteriordb_path,
            "rl_algorithm": rl_algorithm,
            "mcmc_env": mcmc_env,
            "repeat_num": repeat_num,
        }
        pdb_run_content = pdb_run_template.render(pdb_run_context)
        pdb_run_path = f"{model_result_dir}/run_pdb_{rl_algorithm}_{mcmc_env}.py"
        with open(pdb_run_path, "w") as file:
            file.write(pdb_run_content)

        bash_template_path = f"{template_root_dir}/template.run-pdb.sh"
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
