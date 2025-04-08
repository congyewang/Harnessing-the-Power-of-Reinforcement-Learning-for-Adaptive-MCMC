from pyrlmala.utils.plot import PlotPipeLine


model_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]

mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

for model_name in model_list:
    model_dir_root_name = model_name.split("-")[1]
    const_dir = f"./{model_dir_root_name}/const"
    for mcmc_env in mcmc_env_list:
        PlotPipeLine().execute(
            mcmc_env=mcmc_env,
            const_dir=const_dir,
            flex_file_path=f"./{model_dir_root_name}/flex/{model_name}_{mcmc_env}_mmd.txt",
            save_path=f"./{model_dir_root_name}/pic/{model_name}_{mcmc_env}_mmd.pdf",
        )
