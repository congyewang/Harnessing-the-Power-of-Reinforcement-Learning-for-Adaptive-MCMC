import numpy as np
import gc
from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
from pyrlmala.utils.posteriordb import PosteriorDBToolbox

model_name = "kidiq-kidscore_momhsiq"
posteriordb_path = "../posteriordb/posterior_database"


pdb_toolbox = PosteriorDBToolbox("../posteriordb/posterior_database")
gs = pdb_toolbox.get_gold_standard(model_name)


sample_dim = 4
initial_sample = gs[0]
initial_step_size = np.array([0.1])
initial_covariance = pdb_toolbox.get_fisher_information_matrix(model_name)
algorithm = "ddpg"
mcmc_env = "mala"
runtime_config_path = f"./config/runtime_{mcmc_env}.toml"

learning_instance = LearningFactory.create_learning_instance(
    algorithm=algorithm,
    model_name=model_name,
    posteriordb_path=posteriordb_path,
    initial_sample=initial_sample,
    initial_covariance=initial_covariance,
    initial_step_size=initial_step_size,
    hyperparameter_config_path=f"./config/{algorithm}_{mcmc_env}.toml",
    actor_config_path="./config/actor.toml",
    critic_config_path="./config/critic.toml",
)


learning_instance.train()


learning_instance.predict()


predicted_sample = learning_instance.predicted_observation[:, 0:sample_dim]

del learning_instance
gc.collect()

mmd = Toolbox.calculate_mmd(gs, predicted_sample)
if mmd:
    print(mmd)
else:
    print("MMD calculation failed.")
