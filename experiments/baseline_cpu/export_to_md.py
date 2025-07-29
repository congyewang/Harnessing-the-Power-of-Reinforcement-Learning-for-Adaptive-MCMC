import pandas as pd
import glob
import re


model_dim = {
    "earnings-earn_height": 3,
    "earnings-log10earn_height": 3,
    "earnings-logearn_height": 3,
    "gp_pois_regr-gp_regr": 3,
    "kidiq-kidscore_momhs": 3,
    "kidiq-kidscore_momiq": 3,
    "kilpisjarvi_mod-kilpisjarvi": 3,
    "mesquite-logmesquite_logvolume": 3,
    "arma-arma11": 4,
    "earnings-logearn_height_male": 4,
    "earnings-logearn_logheight_male": 4,
    "garch-garch11": 4,
    "hmm_example-hmm_example": 4,
    "kidiq-kidscore_momhsiq": 4,
    "one_comp_mm_elim_abs-one_comp_mm_elim_abs": 4,
    "earnings-logearn_interaction": 5,
    "earnings-logearn_interaction_z": 5,
    "kidiq-kidscore_interaction": 5,
    "kidiq_with_mom_work-kidscore_interaction_c": 5,
    "kidiq_with_mom_work-kidscore_interaction_c2": 5,
    "kidiq_with_mom_work-kidscore_interaction_z": 5,
    "kidiq_with_mom_work-kidscore_mom_work": 5,
    "low_dim_gauss_mix-low_dim_gauss_mix": 5,
    "mesquite-logmesquite_logva": 5,
    "bball_drive_event_0-hmm_drive_0": 6,
    "bball_drive_event_1-hmm_drive_1": 6,
    "sblrc-blr": 6,
    "sblri-blr": 6,
    "arK-arK": 7,
    "mesquite-logmesquite_logvash": 7,
    "hudson_lynx_hare-lotka_volterra": 8,
    "mesquite-logmesquite": 8,
    "mesquite-logmesquite_logvas": 8,
    "mesquite-mesquite": 8,
    "eight_schools-eight_schools_centered": 10,
    "eight_schools-eight_schools_noncentered": 10,
    "nes1972-nes": 10,
    "nes1976-nes": 10,
    "nes1980-nes": 10,
    "nes1984-nes": 10,
    "nes1988-nes": 10,
    "nes1992-nes": 10,
    "nes1996-nes": 10,
    "nes2000-nes": 10,
    "gp_pois_regr-gp_pois_regr": 13,
    "diamonds-diamonds": 26,
    "mcycle_gp-accel_gp": 66,
}


file_path = glob.glob("./results/*.csv")


df_model_dim = pd.DataFrame(list(model_dim.items()), columns=["Model", "Dimension"])
print(df_model_dim.head())


cpu_time = {}
for i in file_path:
    df = pd.read_csv(i)
    model_name = re.search(r"baseline_(.+).csv", i).group(1)
    if df.shape[0] == 0:
        continue
    else:
        cpu_time[model_name] = df["cpu_time"].mean().item()


df_model_dim["cpu_time"] = df_model_dim["Model"].map(cpu_time)
df_sorted = df_model_dim.sort_values(by=["Dimension", "Model"], ascending=[True, True])


df_sorted.to_markdown("baseline_cpu_results.md")
