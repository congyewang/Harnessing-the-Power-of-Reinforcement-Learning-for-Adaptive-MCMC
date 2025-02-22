import os
import jinja2
import shutil


class BenchmarkFactory:
    """
    Factory class to generate benchmark files for MCMC algorithms.

    Attributes:
        _model_list (List[str]): The list of model names.
        _mcmc_envs (Dict[str, str]): The dictionary of MCMC environments.
        _random_seed (int): The random seed.
        _step_size_list (List[float]): The list of step sizes.
    """

    _model_list = ["gaussian", "laplace_1", "laplace_2", "laplace_4"]
    _mcmc_envs = {
        # "mala": "MALAEnv",
        "barker": "BarkerEnv",
    }
    _random_seed = 42
    _step_size_list = [0.5 * (i + 1) for i in range(20)]

    @staticmethod
    def rander_template(template_dir_path: str, output_path: str, **kwargs) -> None:
        """
        Render Jinja2 template and write to output file.

        Args:
            template_dir_path (str): The path to the directory containing the Jinja2 template.
            output_path (str): The path to the output file.
            **kwargs: The arguments to pass to the Jinja2 template.
        """
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(template_dir_path))
        template = env.get_template("template.mcmc_const.py")
        output = template.render(**kwargs)

        with open(output_path, "w") as f:
            f.write(output)

    @staticmethod
    def execute(
        root_dir: str = ".",
        template_relative_dir: str = os.path.join("template"),
    ) -> None:
        sh_scripts = []
        for model_name in BenchmarkFactory._model_list:
            for mcmc_env_name, mcmc_env in BenchmarkFactory._mcmc_envs.items():
                for step_size in BenchmarkFactory._step_size_list:
                    # Create output directory if not exists
                    output_dir = os.path.join(
                        root_dir,
                        model_name,
                        mcmc_env_name,
                        f"{mcmc_env_name}_{str(step_size).replace('.', '_')}",
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    BenchmarkFactory.rander_template(
                        template_dir_path=os.path.join(root_dir, template_relative_dir),
                        output_path=os.path.join(output_dir, "mcmc_const.py"),
                        random_seed=BenchmarkFactory._random_seed,
                        step_size=step_size,
                        env_name=mcmc_env,
                    )

                    sh_script_path = os.path.join(output_dir, "run-mcmc.sh")
                    shutil.copy(
                        os.path.join(
                            root_dir,
                            template_relative_dir,
                            "template.run-mcmc_const.sh",
                        ),
                        sh_script_path,
                    )
                    sh_scripts.append(sh_script_path)

        # Create sbatch script
        sbatch_script_path = os.path.join(root_dir, "submit_all.sh")
        with open(sbatch_script_path, "w+") as sbatch_file:
            sbatch_file.write("#!/bin/bash\n\n")
            for sh_script in sh_scripts:
                sbatch_file.write(f"sbatch {sh_script}\n")

        # Make sbatch script executable
        os.chmod(sbatch_script_path, 0o755)
