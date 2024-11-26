from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """
    Experiment configuration data class.

    Attributes:
        exp_name (str): Name of the experiment.
        seed (int): Random seed.
        torch_deterministic (bool): Flag to set the PyTorch deterministic mode.
        cuda (bool): Flag to enable CUDA.
        track (bool): Flag to enable tracking.
        wandb_project_name (Optional[str]): WandB project name.
        wandb_entity (Optional[str]): WandB entity.
        capture_video (bool): Flag to enable video capturing.
        save_model (bool): Flag to save the model.
        upload_model (bool): Flag to upload the model.
        hf_entity (Optional[str]): Hugging Face entity.
    """

    exp_name: str
    seed: int
    torch_deterministic: bool
    cuda: bool
    track: bool
    wandb_project_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: Optional[str] = None


@dataclass
class AlgorithmGeneralConfig:
    """
    General configuration data class for the algorithm.

    Attributes:
        env_id (str): Environment ID.
        total_timesteps (int): Total timesteps.
        predicted_timesteps (int): Predicted timesteps.
        learning_starts (int): Learning starts.
        buffer_size (int): Buffer size.
        batch_size (int): Batch size.
        gamma (float): Gamma.
        actor_learning_rate (float): Actor learning rate.
        critic_learning_rate (float): Critic learning rate.
        actor_gradient_clipping (float): Actor gradient clipping.
        critic_gradient_clipping (float): Critic gradient clipping.
    """

    env_id: str
    total_timesteps: int
    predicted_timesteps: int
    learning_starts: int
    buffer_size: int
    batch_size: int
    gamma: float
    actor_learning_rate: float
    critic_learning_rate: float
    actor_gradient_clipping: float
    critic_gradient_clipping: float


@dataclass
class AlgorithmSpecificConfig:
    """
    Specific configuration data class for the algorithm.

    Attributes:
        tau (float): Tau.
        policy_noise (float): Policy noise.
        exploration_noise (float): Exploration noise.
        policy_frequency (int): Policy frequency.
        noise_clip (float): Noise clip.
    """

    tau: float
    policy_noise: float
    exploration_noise: float
    policy_frequency: int
    noise_clip: float


@dataclass
class AlgorithmConfig:
    """
    Algorithm configuration data class.

    Attributes:
        general: AlgorithmGeneralConfig.
        specific: AlgorithmSpecificConfig.
    """

    general: AlgorithmGeneralConfig
    specific: AlgorithmSpecificConfig


@dataclass
class Config:
    """
    Configuration data class.

    Attributes:
        experiments: ExperimentConfig.
        algorithm: AlgorithmConfig.
    """

    experiments: ExperimentConfig
    algorithm: AlgorithmConfig


@dataclass
class PolicyNetworkConfig:
    """
    Configuration data class for the policy network.

    Attributes:
        hidden_layers (List[int]): Hidden layers.
        activation_function (str): Activation function.
    """

    hidden_layers: List[int]
    activation_function: str


@dataclass
class QNetworkConfig:
    """
    Configuration data class for the Q network.

    Attributes:
        hidden_layers (List[int]): Hidden layers.
        activation_function (str): Activation function.
    """

    hidden_layers: List[int]
    activation_function: str
