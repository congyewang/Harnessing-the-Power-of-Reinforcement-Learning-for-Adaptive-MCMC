from pyrlmala.utils.benchmark import MMDBatchRunner


def main(model_name: str):
    mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]
    step_size_list = list(range(1, 20, 1)) + [0.1, 0.5]

    for mcmc_env in mcmc_env_list:
        for step_size in step_size_list:
            const_batch_run = MMDBatchRunner(
                model_name=model_name,
                posteriordb_path="../posteriordb/posterior_database",
            )
            const_batch_run.run(
                mcmc_env=mcmc_env,
                step_size=step_size,
                repeat_count=10,
                save_root_path=".",
            )


if __name__ == "__main__":
    model_name = "test-laplace_4-test-laplace_4"
    main(model_name)
