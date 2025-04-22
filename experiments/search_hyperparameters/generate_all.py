import jinja2
from pyrlmala.utils import Toolbox


hyperparameter_template_path = "./templete/config_template.toml"
py_template_path = "./templete/search_run.py"
bash_template_path = "./templete/run-search.sh"
exp_name = "search_hyperparameters"
random_seed = 0
env_id = "MALAEnv-v1.0"

critic_learning_rate_list = [10.0, 1.0, 0.1, 0.01, 0.001]
actor_learning_rate_multiplier_list = [0.1, 0.01, 0.001]
tau_list = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

with open(hyperparameter_template_path, "r") as hyperparameter_file:
    hyperparameter_template = jinja2.Template(hyperparameter_file.read())

with open(py_template_path, "r") as py_file:
    py_template = jinja2.Template(py_file.read())

with open(bash_template_path, "r") as bash_file:
    bash_template = jinja2.Template(bash_file.read())

for tau in tau_list:
    for critic_learning_rate in critic_learning_rate_list:
        for actor_learning_rate_multiplier in actor_learning_rate_multiplier_list:
            actor_learning_rate = actor_learning_rate_multiplier * critic_learning_rate

            # Generate the configuration file
            hyperparameter_context = {
                "exp_name": exp_name,
                "random_seed": random_seed,
                "env_id": env_id,
                "critic_learning_rate": critic_learning_rate,
                "actor_learning_rate": actor_learning_rate,
                "tau": tau,
            }
            config_content = hyperparameter_template.render(hyperparameter_context)

            config_output_path = f"./config/ddpg/ddpg_mala_{critic_learning_rate}_{actor_learning_rate}_{tau}.toml"
            Toolbox.create_folder(config_output_path)

            with open(config_output_path, "w") as file:
                file.write(config_content)

            # Generate the Python script
            py_context = {
                "critic_learning_rate": critic_learning_rate,
                "actor_learning_rate": actor_learning_rate,
                "tau": tau,
            }
            py_content = py_template.render(py_context)
            py_output_path = f"./search_run_{critic_learning_rate}_{actor_learning_rate}_{tau}.py"

            with open(py_output_path, "w") as file:
                file.write(py_content)

            # Generate the shell script
            bash_context = {
                "critic_learning_rate": critic_learning_rate,
                "actor_learning_rate": actor_learning_rate,
                "tau": tau,
            }
            bash_content = bash_template.render(bash_context)
            bash_output_path = f"./run-search_{critic_learning_rate}_{actor_learning_rate}_{tau}.sh"
            with open(bash_output_path, "w") as file:
                file.write(bash_content)
