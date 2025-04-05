from pyrlmala.utils.benchmark import MMDBatchRunner


def main(model_name: str, step_size: float) -> None:
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

    for mcmc_env in mcmc_env_list:
        const_batch_run = MMDBatchRunner(
            model_name=model_name,
            posteriordb_path="../../posteriordb/posterior_database",
        )
        const_batch_run.run(
            mcmc_env=mcmc_env,
            step_size=step_size,
            repeat_count=10,
            save_root_path=".",
        )


if __name__ == "__main__":
    main(model_name="{{ model_name }}", step_size={{ step_size }})
