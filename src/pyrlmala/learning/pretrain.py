from typing import Tuple, TypeVar

import numpy as np
import torch
import torch.optim as optim
from jaxtyping import Float
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..agent import PolicyNetwork, QNetwork
from ..envs import MCMCEnvBase

T = TypeVar("T")


class PretrainMockDataset(Dataset[T]):
    """
    Pretrain Mock Dataset for Actor.

    Attributes:
        initial_sample (Float[torch.Tensor, "initial sample"]): Initial sample.
        sample_dim (int): Sample dimension.
        step_size (float): Step size.
        num (int): Number of samples.
        mock_state (Float[torch.Tensor, "state"]): Mock state.
        mock_step_size (Float[torch.Tensor, "step size"]): Mock step size.
    """

    def __init__(
        self,
        initial_sample: Float[torch.Tensor, "initial sample"],
        step_size: float = 1.0,
        num_data: int = 1_000,
        mag: float = 10.0,
    ) -> None:
        """
        Pretrain Mock Dataset for Actor.

        Args:
            initial_sample (Float[torch.Tensor, "initial sample"]): Initial sample.
            step_size (float, optional): Step size. Defaults to 1.0.
            num_data (int, optional): Number of samples. Defaults to 1_000.
            mag (float, optional): Magnification. Defaults to 10.0.

        Raises:
            ValueError: If step size is non-positive.
            ValueError: If number of samples is non-positive.
            ValueError: If magnification is non-positive.
        """
        if step_size <= 0:
            raise ValueError("Step size must be positive.")
        if num_data <= 0 and not isinstance(num_data, int):
            raise ValueError("Number of samples must be a positive integer.")
        if mag <= 0:
            raise ValueError("Magnification must be positive.")

        self.initial_sample = initial_sample
        self.sample_dim = len(initial_sample)
        self.step_size = step_size
        self.num_data = num_data
        self.mag = mag

        self.mock_state = self.mock_state_generator()
        self.mock_step_size = self.mock_step_size_generator()

    def __len__(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: Number of samples.
        """
        return self.num_data

    def __getitem__(
        self, index: int
    ) -> Tuple[Float[torch.Tensor, "state"], Float[torch.Tensor, "action"]]:
        """
        Returns the state and action at the given index.

        Returns:
            Tuple[Float[torch.Tensor, "state"], Float[torch.Tensor, "action"]]: State and action.
        """
        return self.mock_state[index].double(), self.mock_step_size[index].double()

    def inverse_softplus(
        self, /, x: Float[torch.Tensor, "input"]
    ) -> Float[torch.Tensor, "output"]:
        """
        Inverse softplus function.

        Returns:
            Float[torch.Tensor, "output"]: Output.
        """
        return x + torch.log1p(-torch.exp(-x))

    def mock_state_generator(self) -> Float[torch.Tensor, "state"]:
        """
        Mock state generator.

        Returns:
            Float[torch.Tensor, "state"]: State.
        """
        multivariate_normal = torch.distributions.MultivariateNormal(
            loc=self.initial_sample,
            covariance_matrix=(
                self.mag * torch.eye(self.sample_dim, dtype=torch.float64)
            ),
        )

        return multivariate_normal.sample((self.num_data, self.sample_dim)).view(
            -1, self.sample_dim << 1
        )

    def mock_step_size_generator(self) -> Float[torch.Tensor, "step size"]:
        """
        Mock step size generator.

        Returns:
            Float[torch.Tensor, "step size"]: Step size.
        """
        return self.inverse_softplus(
            torch.repeat_interleave(
                torch.tensor([self.step_size]), self.num_data << 1
            ).view(self.num_data, -1)
        )


class PretrainFactory:
    @staticmethod
    def train(
        actor: PolicyNetwork,
        initial_sample: Float[torch.Tensor, "initial sample"],
        step_size: float = 1.0,
        num_data: int = 1_000,
        mag: float = 10.0,
        num_epochs: int = 100,
        batch_size: int = 16,
        shuffle: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        verbose: bool = True,
    ) -> PolicyNetwork:
        """
        Pretrain Actor. This method is used to pretrain the actor using a mock dataset.

        Args:
            actor (PolicyNetwork): Actor.
            initial_sample (Float[torch.Tensor, "initial sample"]): Initial sample.
            step_size (float, optional): Step size. Defaults to 1.0.
            num_data (int, optional): Number of samples. Defaults to 1_000.
            mag (float, optional): Magnification. Defaults to 10.0.
            num_epochs (int, optional): Number of epochs. Defaults to 100.
            batch_size (int, optional): Batch size. Defaults to 16.
            shuffle (bool, optional): Shuffle. Defaults to True.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            verbose (bool, optional): Verbose. Defaults to False.

        Returns:
            PolicyNetwork: Actor.
        """
        # Load Mock Dataset
        mock_dataset = PretrainMockDataset(initial_sample, step_size, num_data, mag)
        data_loader = DataLoader(mock_dataset, batch_size=batch_size, shuffle=shuffle)

        # Configure Actor
        actor.train()
        actor.to(device)

        # Loss function and Optimizer
        criterion = nn.MSELoss()
        optimizer = optim.SGD(actor.parameters(), lr=0.01)

        # Progress Bar
        progress_bar = tqdm(
            range(num_epochs), desc="Training Epochs", disable=not verbose, leave=False
        )

        # Training Process
        for epoch in progress_bar:
            for _, (inputs, targets) in enumerate(data_loader):
                # Forward pass
                predictions = actor(inputs)
                loss = criterion(predictions, targets)

                # Backward Pass and Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Optionally add progress bar updates with detailed logging
            if (epoch + 1) % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return actor


class PretrainingQNetwork:
    def __init__(
        self,
        mock_env: MCMCEnvBase,
        q_network: QNetwork,
        initial_sample: Float[torch.Tensor, "initial sample"],
        step_size: float = 1.0,
        mag: float = 10.0,
        batch_size: int = 64,
        pretrain_steps: int = 1_000,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        verbose: bool = True,
    ) -> None:
        """
        Pretraining class for the DDPG Q-function.

        Args:
            mock_env (MCMCEnvBase): Mock environment.
            q_network (QNetwork): Q-network.
            initial_sample (Float[torch.Tensor, "initial sample"]): Initial sample.
            step_size (float, optional): Step size. Defaults to 1.0.
            mag (float, optional): Magnification. Defaults to 10.0.
            batch_size (int, optional): Batch size. Defaults to 64.
            pretrain_steps (int, optional): Number of pretraining steps. Defaults to 1_000.
            device (torch.device, optional): Device. Defaults to torch.device("cpu").
            verbose (bool, optional): Verbose. Defaults to True.
        """
        self.mock_env = mock_env
        self.q_network = q_network
        self.q_network.to(device)
        self.initial_sample = initial_sample
        self.step_size = step_size
        self.mag = mag
        self.verbose = verbose

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.01)
        self.batch_size = batch_size
        self.pretrain_steps = pretrain_steps

    def generate_pretrain_data(
        self,
    ) -> Tuple[
        Float[torch.Tensor, "observation"],
        Float[torch.Tensor, "action"],
        Float[torch.Tensor, "q value"],
    ]:
        """
        Generate pretraining data by sampling observations and actions.

        Returns:
            Tuple[
                Float[torch.Tensor, "observation"],
                Float[torch.Tensor, "action"],
                Float[torch.Tensor, "q value"]
            ]: Observations, actions, and Q-values.
        """
        observations = []
        actions = []
        q_values = []

        for _ in range(self.batch_size):
            # Sample observation ~ N(initial_sample, 10I)
            observation = self.mock_env.np_random.multivariate_normal(
                mean=self.mock_env.initial_sample,
                cov=self.mag * np.eye(self.mock_env.sample_dim),
            )

            # Sample action ~ N(initial_step_size, 10)
            action = self.mock_env.np_random.normal(
                loc=self.mock_env.initial_step_size,
                scale=np.sqrt(self.mag),
                size=2,
            )

            # Get the Q-value for the observation-action pair
            obs_tensor = torch.from_numpy(
                np.concatenate([observation, observation])
            ).float()
            action_tensor = torch.from_numpy(action).float()
            q_value = (
                self.q_network(obs_tensor.unsqueeze(0), action_tensor.unsqueeze(0))
                .detach()
                .item()
            )

            observations.append(obs_tensor)
            actions.append(action_tensor)
            q_values.append(q_value)

        return (
            torch.stack(observations),
            torch.stack(actions),
            torch.tensor(q_values).float(),
        )

    def train_one_step(
        self,
        observations: Float[torch.Tensor, "observations"],
        actions: Float[torch.Tensor, "actions"],
        q_values: Float[torch.Tensor, "q values"],
    ) -> float:
        """
        Perform one step of Q-network optimization.

        Args:
            observations (Float[torch.Tensor, "observations"]): Observations.
            actions (Float[torch.Tensor, "actions"]): Actions.
            q_values (Float[torch.Tensor, "q values"]): Q-values.
        """
        # Compute predicted Q-values
        predicted_q_values = self.q_network(observations, actions).squeeze()

        # Compute loss (MSE)
        loss = torch.nn.functional.mse_loss(predicted_q_values, q_values)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self) -> None:
        """
        Run the pretraining process for the specified number of steps.
        """
        # Progress Bar
        progress_bar = tqdm(
            range(self.pretrain_steps),
            desc="Training Epochs",
            disable=not self.verbose,
            leave=False,
        )

        # for step in range(self.pretrain_steps):
        for step in progress_bar:
            # Generate batch of data
            observations, actions, q_values = self.generate_pretrain_data()

            # Train for one step
            loss = self.train_one_step(observations, actions, q_values)

            # Optionally add progress bar updates with detailed logging
            if (step + 1) % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss:.4f}"})
