Examples
========

This page contains examples of how to use the PyRLMala library.

.. code-block:: python
   :caption: Example of Sampling from Stan Model using DDPG
   :linenos:

   import json
   from functools import partial

   import matplotlib.pyplot as plt
   import numpy as np
   import torch
   from toolz import pipe

   from pyrlmala.learning import LearningFactory
   from pyrlmala.utils import Toolbox

   # Fix steps per episode to 100
   plot_agent_100 = partial(Toolbox.plot_agent, steps_per_episode=100)

   # Make Log Target PDF and Grad Log Target PDF
   stan_code_path = "banana.stan"
   stan_data_path = "banana.json"

   with open(stan_data_path, "r") as f:
      data = json.load(f)

      log_target_pdf = Toolbox.make_log_target_pdf(stan_code_path, data)
      grad_log_target_pdf = Toolbox.make_grad_log_target_pdf(stan_code_path, data)

   # Generate Learning Instance from LearningFactory
   sample_dim = 2
   initial_sample = np.zeros(sample_dim)
   algorithm = "ddpg" # or "td3"

   learning_instance = LearningFactory.create_learning_instance(
      algorithm=algorithm,
      log_target_pdf=log_target_pdf,
      grad_log_target_pdf=grad_log_target_pdf,
      initial_sample=initial_sample,
      hyperparameter_config_path="../../config/ddpg.toml",
      actor_config_path="../../config/actor.toml",
      critic_config_path="../../config/critic.toml",
   )

   # Train Learning Instance
   learning_instance.train()

   # Prediction from Trained Learning Instance
   learning_instance.predict()

   # Trace Plot
   accepted_sample = getattr(learning_instance, "env").envs[0].unwrapped.store_accepted_sample

   plt.plot(accepted_sample[:, 0], accepted_sample[:, 1], 'o-', alpha=0.1)
   plt.show()

   # Policy Plot
   x_range = (-3, 3, 0.1)
   y_range = (5, 20, 1)
   policy = lambda x: learning_instance.actor(x.double())

   Toolbox.policy_plot_2D_heatmap(policy, torch.arange(*x_range), torch.arange(*y_range))

   # Critic Values Plot
   pipe(learning_instance, lambda x: getattr(x, "critic_values"), plot_agent_100)

   # Critic Loss Plot
   pipe(learning_instance, lambda x: getattr(x, "critic_loss"), plot_agent_100)

   # Actor Loss Plot
   pipe(learning_instance, lambda x: getattr(x, "actor_loss"), plot_agent_100)
