from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ExperimentConfig:
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
    env_id: str
    total_timesteps: int
    learning_starts: int
    learning_rate: float
    buffer_size: int
    batch_size: int
    gamma: float


@dataclass
class AlgorithmDDPGConfig:
    tau: float
    policy_noise: float
    exploration_noise: float
    policy_frequency: int
    noise_clip: float


@dataclass
class AlgorithmConfig:
    general: AlgorithmGeneralConfig
    ddpg: AlgorithmDDPGConfig


@dataclass
class Config:
    experiments: ExperimentConfig
    algorithm: AlgorithmConfig


@dataclass
class PolicyNetworkConfig:
    hidden_layers: List[int]
    activation_function: str


@dataclass
class QNetworkConfig:
    hidden_layers: List[int]
    activation_function: str
