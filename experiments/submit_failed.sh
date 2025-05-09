#!/bin/bash

cd whole_results

cd gp_pois_regr-gp_pois_regr
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd gp_pois_regr-gp_pois_regr
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd gp_pois_regr-gp_pois_regr
sbatch run_bash_ddpg_barker.sh
cd -

cd gp_pois_regr-gp_pois_regr
sbatch run_bash_ddpg_mala.sh
cd -

cd kidiq-kidscore_interaction
sbatch run_bash_ddpg_barker.sh
cd -

cd kidiq-kidscore_interaction
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd kidiq-kidscore_interaction
sbatch run_bash_ddpg_mala.sh
cd -

cd kidiq-kidscore_interaction
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd mesquite-mesquite
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd mesquite-mesquite
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd mesquite-mesquite
sbatch run_bash_ddpg_barker.sh
cd -

cd mesquite-mesquite
sbatch run_bash_ddpg_mala.sh
cd -

cd sblrc-blr
sbatch run_bash_ddpg_barker.sh
cd -

cd kidiq-kidscore_momhsiq
sbatch run_bash_ddpg_barker.sh
cd -

cd sblri-blr
sbatch run_bash_ddpg_barker.sh
cd -

cd kilpisjarvi_mod-kilpisjarvi
sbatch run_bash_ddpg_barker.sh
cd -

cd eight_schools-eight_schools_noncentered
sbatch run_bash_ddpg_barker.sh
cd -

cd mcycle_gp-accel_gp
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd mcycle_gp-accel_gp
sbatch run_bash_ddpg_mala.sh
cd -

cd mcycle_gp-accel_gp
sbatch run_bash_ddpg_barker.sh
cd -

cd mcycle_gp-accel_gp
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd bball_drive_event_1-hmm_drive_1
sbatch run_bash_ddpg_barker.sh
cd -

cd diamonds-diamonds
sbatch run_bash_ddpg_barker.sh
cd -

cd eight_schools-eight_schools_centered
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd eight_schools-eight_schools_centered
sbatch run_bash_ddpg_mala.sh
cd -

cd eight_schools-eight_schools_centered
sbatch run_bash_ddpg_barker.sh
cd -

cd eight_schools-eight_schools_centered
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd nes1988-nes
sbatch run_bash_ddpg_barker.sh
cd -

cd hudson_lynx_hare-lotka_volterra
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd hudson_lynx_hare-lotka_volterra
sbatch run_bash_ddpg_mala.sh
cd -

cd hudson_lynx_hare-lotka_volterra
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd hudson_lynx_hare-lotka_volterra
sbatch run_bash_ddpg_barker.sh
cd -

cd nes1996-nes
sbatch run_bash_ddpg_barker.sh
cd -

cd earnings-earn_height
sbatch run_bash_ddpg_barker.sh
cd -

cd earnings-earn_height
sbatch run_bash_ddpg_mala.sh
cd -

cd earnings-earn_height
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd earnings-earn_height
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd arma-arma11
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd arma-arma11
sbatch run_bash_ddpg_mala_esjd.sh
cd -

cd earnings-logearn_interaction
sbatch run_bash_ddpg_barker.sh
cd -

cd one_comp_mm_elim_abs-one_comp_mm_elim_abs
sbatch run_bash_ddpg_barker_esjd.sh
cd -

cd earnings-log10earn_height
sbatch run_bash_ddpg_barker.sh
cd -

