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
    if model_name == "kidiq_with_mom_work-kidscore_mom_work":
        step_size = 2.96e-02
    elif model_name == "bball_drive_event_0-hmm_drive_0":
        step_size = 3.36e-02
    elif model_name == "kilpisjarvi_mod-kilpisjarvi":
        step_size = 3.75e-02
    elif model_name == "kidiq-kidscore_momhsiq":
        step_size = 6.66e-03
    elif model_name == "eight_schools-eight_schools_noncentered":
        step_size = 5.64e-01
    elif model_name == "nes2000-nes":
        step_size = 8.59e-04
    elif model_name == "nes1996-nes":
        step_size = 2.91e-06
    elif model_name == "earnings-logearn_interaction":
        step_size = 3.06e-02
    elif model_name == "arK-arK":
        step_size = 4.37e-05
    elif model_name == "gp_pois_regr-gp_regr":
        step_size = 8.78e-02
    elif model_name == "diamonds-diamonds":
        step_size = 1.02e-07
    elif model_name == "garch-garch11":
        step_size = 2.00e-03
    elif model_name == "mesquite-logmesquite":
        step_size = 3.29e-04
    elif model_name == "mesquite-logmesquite_logva":
        step_size = 2.15e-02
    elif model_name == "nes1992-nes":
        step_size = 1.08e-07
    elif model_name == "nes1976-nes":
        step_size = 9.61e-08
    elif model_name == "sblri-blr":
        step_size = 1.41e-02
    elif model_name == "earnings-log10earn_height":
        step_size = 7.81e-02
    elif model_name == "earnings-logearn_height_male":
        step_size = 7.74e-03
    elif model_name == "nes1988-nes":
        step_size = 1.04e-07
    elif model_name == "mesquite-logmesquite_logvas":
        step_size = 2.56e-04
    elif model_name == "earnings-earn_height":
        step_size = 1.70e-05
    elif model_name == "mcycle_gp-accel_gp":
        step_size = 5.64e-01
    elif model_name == "hmm_example-hmm_example":
        step_size = 4.26e-03
    elif model_name == "bball_drive_event_1-hmm_drive_1":
        step_size = 1.38e-02
    elif model_name == "nes1980-nes":
        step_size = 8.29e-04
    elif model_name == "earnings-logearn_height":
        step_size = 7.24e-02
    elif model_name == "one_comp_mm_elim_abs-one_comp_mm_elim_abs":
        step_size = 1.37e-06
    elif model_name == "kidiq-kidscore_momhs":
        step_size = 8.16e-02
    elif model_name == "earnings-logearn_logheight_male":
        step_size = 2.67e-02
    elif model_name == "kidiq-kidscore_momiq":
        step_size = 1.71e-01
    elif model_name == "hudson_lynx_hare-lotka_volterra":
        step_size = 1.93e-04
    elif model_name == "nes1972-nes":
        step_size = 1.02e-07
    elif model_name == "sblrc-blr":
        step_size = 2.68e-05
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_c":
        step_size = 3.08e-02
    elif model_name == "nes1984-nes":
        step_size = 8.99e-08
    elif model_name == "eight_schools-eight_schools_centered":
        step_size = 6.33e-08
    elif model_name == "earnings-logearn_interaction_z":
        step_size = 3.18e-02
    elif model_name == "mesquite-logmesquite_logvash":
        step_size = 1.91e-05
    elif model_name == "mesquite-logmesquite_logvolume":
        step_size = 4.36e-02
    elif model_name == "kidiq-kidscore_interaction":
        step_size = 3.07e-02
    elif model_name == "arma-arma11":
        step_size = 5.06e-03
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_c2":
        step_size = 1.24e-03
    elif model_name == "kidiq_with_mom_work-kidscore_interaction_z":
        step_size = 3.05e-02
    elif model_name == "low_dim_gauss_mix-low_dim_gauss_mix":
        step_size = 2.95e-02
    elif model_name == "gp_pois_regr-gp_pois_regr":
        step_size = 5.64e-01
    elif model_name == "mesquite-mesquite":
        step_size = 3.28e-04
    else:
        step_size = 1.0e-2

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
                "exploration_noise": str(0.2 * output_initial_step_size(model_name)),
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
