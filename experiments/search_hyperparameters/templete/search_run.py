from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from cytoolz import pipe

from pyrlmala.learning import LearningFactory
from pyrlmala.utils import Toolbox
from pyrlmala.utils.plot import GeneralPlot

critic_learning_rate = {{ critic_learning_rate }}
actor_learning_rate = {{ actor_learning_rate }}
tau = {{ tau }}

plot_agent_500 = partial(GeneralPlot.plot_agent, steps_per_episode=500)

model_name = "test-banana-test-banana"
posteriordb_path = "../posteriordb/posterior_database"


sample_dim = 2
initial_sample = np.array([0.0, 10.0])
initial_step_size = np.array([0.2])
algorithm = "ddpg"
mcmc_env = "mala"

learning_instance = LearningFactory.create_learning_instance(
    algorithm=algorithm,
    model_name=model_name,
    posteriordb_path=posteriordb_path,
    initial_sample=initial_sample,
    initial_step_size=initial_step_size,
    hyperparameter_config_path=f"./config/ddpg/ddpg_mala_{critic_learning_rate}_{actor_learning_rate}_{tau}.toml",
    actor_config_path="./templete/actor.toml",
    critic_config_path="./templete/critic.toml",
)


learning_instance.train()

x_range = (-4, 4, 0.1)
y_range = (5, 15, 0.1)
policy = lambda x: learning_instance.actor(x.double())

plt.figure()
policy_save_path = f"pic/policy/policy_plot_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(policy_save_path)
GeneralPlot.policy_plot_2D_heatmap(
    policy,
    torch.arange(*x_range),
    torch.arange(*y_range),
    save_path=policy_save_path,
    title="Policy Plot",
)


lower_window_size = 0
upper_window_size = learning_instance.env.envs[0].get_wrapper_attr("current_step") - 1
trace_save_path = f"pic/trace/trace_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(trace_save_path)
accepted_sample = pipe(
    learning_instance,
    lambda x: getattr(x, "env"),
    lambda x: x.get_attr("store_accepted_sample"),
)[0]

plt.figure()
plt.plot(
    accepted_sample[lower_window_size:upper_window_size, 0],
    accepted_sample[lower_window_size:upper_window_size, 1],
    "o-",
    alpha=0.1,
)
plt.savefig(trace_save_path)

plt.figure()
critic_values_save_path = f"pic/critic_values/critic_values_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(critic_values_save_path)
pipe(
    learning_instance,
    lambda x: getattr(x, "critic_values"),
    lambda x: plot_agent_500(
        x,
        title="Critic Values",
        save_path=critic_values_save_path,
    ),
)

plt.figure()
critic_loss_save_path = f"pic/critic_loss/critic_loss_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(critic_loss_save_path)
pipe(
    learning_instance,
    lambda x: getattr(x, "critic_loss"),
    lambda x: plot_agent_500(
        x,
        title="Critic Loss",
        save_path=critic_loss_save_path,
    ),
)

plt.figure()
actor_loss_save_path = f"pic/actor_loss/actor_loss_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(actor_loss_save_path)
pipe(
    learning_instance,
    lambda x: getattr(x, "actor_loss"),
    lambda x: plot_agent_500(
        x,
        title="Actor Loss",
        save_path=actor_loss_save_path,
    ),
)

plt.figure()
reward_save_path = f"pic/reward/reward_{critic_learning_rate}_{actor_learning_rate}_{tau}.png"
Toolbox.create_folder(reward_save_path)
reward = pipe(
    learning_instance,
    lambda x: getattr(x, "env"),
    lambda x: x.get_attr("store_reward")[0],
    lambda x: GeneralPlot.reward_plot(
        x,
        title="Reward Plot",
        save_path=reward_save_path,
    ),
)
