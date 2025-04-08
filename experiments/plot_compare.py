from pyrlmala.utils.plot import PlotPipeLine


model_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]

mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

bootstrap_res = {
    model_list[0]: (0.00011220981392776608, 3.377862281415175e-05),
    model_list[1]: (0.00010787792317374922, 4.735600841900292e-05),
    model_list[2]: (3.278534342534556e-05, 7.637442387818593e-06),
}

for model_name in model_list:
    model_dir_root_name = model_name.split("-")[1]
    const_dir = f"./{model_dir_root_name}/const"
    for mcmc_env in mcmc_env_list:
        PlotPipeLine().execute(
            mcmc_env=mcmc_env,
            const_dir=const_dir,
            flex_file_path=f"./{model_dir_root_name}/flex/{model_name}_{mcmc_env}_mmd.txt",
            bootstrap_tuple=bootstrap_res[model_name],
            save_path=f"./{model_dir_root_name}/pic/{model_name}_{mcmc_env}_mmd.pdf",
        )
