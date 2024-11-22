import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
from scipy.stats import multivariate_normal
from stable_baselines3.common.buffers import ReplayBuffer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.pyrlmala.agent import PolicyNetwork, QNetwork
from src.pyrlmala.config import (
    HyperparameterConfigParser,
    PolicyNetworkConfigParser,
    QNetworkConfigParser,
)
from src.pyrlmala.config.config_parser import (
    HyperparameterConfigParser,
    PolicyNetworkConfigParser,
    QNetworkConfigParser,
)
from src.pyrlmala.envs import BarkerEnv
from src.pyrlmala.learning import LearningDDPG, LearningTD3
from src.pyrlmala.utils import Toolbox


# Environment
env_id = "BarkerEnv-v1.0"
sample_dim = 2

log_target_pdf = lambda x: multivariate_normal.logpdf(
    x, mean=np.zeros(sample_dim), cov=np.eye(sample_dim)
)
grad_log_target_pdf = lambda x: -x

initial_sample = np.zeros(sample_dim)

# Configurations
args = HyperparameterConfigParser(config_file="src/pyrlmala/config/default/ddpg.toml")
actor_config = PolicyNetworkConfigParser(
    config_file="src/pyrlmala/config/default/actor.toml"
)
critic_config = QNetworkConfigParser(
    config_file="src/pyrlmala/config/default/critic.toml"
)

# Instantiate
envs = Toolbox.make_env(
    env_id, log_target_pdf, grad_log_target_pdf, initial_sample, total_timesteps=1000
)
env = gym.vector.SyncVectorEnv([envs])
predicted_env = gym.vector.SyncVectorEnv([envs])
actor = PolicyNetwork(env, actor_config).double()
target_actor = PolicyNetwork(env, actor_config).double()
critic = QNetwork(env, critic_config).double()
target_critic = QNetwork(env, critic_config).double()
critic2 = QNetwork(env, critic_config).double()
target_critic2 = QNetwork(env, critic_config).double()
actor_optimizer = optim.Adam(
    list(actor.parameters()), lr=args.algorithm.general.learning_rate
)
critic_optimizer = optim.Adam(
    list(actor.parameters()), lr=args.algorithm.general.learning_rate
)
replay_buffer = ReplayBuffer(
    args.algorithm.general.buffer_size,
    env.single_observation_space,
    env.single_action_space,
    torch.device("cpu"),
    handle_timeout_termination=False,
)

# Learning Instance
ddpg_learning = LearningDDPG(
    env,
    actor,
    target_actor,
    critic,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    replay_buffer,
)

td3_learning = LearningTD3(
    env,
    actor,
    target_actor,
    critic,
    target_critic,
    critic2,
    target_critic2,
    actor_optimizer,
    critic_optimizer,
    replay_buffer,
)

# Learning
# ddpg_learning.train()
# ddpg_learning.predict(predicted_env)
td3_learning.train()
td3_learning.predict(predicted_env)
