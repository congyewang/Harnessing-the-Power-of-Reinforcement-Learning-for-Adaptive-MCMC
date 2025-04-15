from pyrlmala.utils.plot import PlotPipeLine


model_list = [f"test-laplace_{i}-test-laplace_{i}" for i in [1, 2, 4]]

mcmc_env_list = ["mala", "mala_esjd", "barker", "barker_esjd"]

bootstrap_res = {
    model_list[0]: (0.016161127676553533,0.016020172963954943,0.016225525627838355),
    model_list[1]: (0.010541329845175418,0.01035008618442601,0.01063463653094654),
    model_list[2]: (0.006906521372515573,0.0068124883746570886,0.006999352326093722),
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
