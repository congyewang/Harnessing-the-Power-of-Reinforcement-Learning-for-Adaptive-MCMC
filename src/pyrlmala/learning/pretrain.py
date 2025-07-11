from abc import abstractmethod
from typing import Any, Tuple

import torch
import torch.optim as optim
from jaxtyping import Float
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..agent import PolicyNetwork
from ..utils import Toolbox


class PretrainDatasetBase(Dataset[Float[torch.Tensor, "num dim"]]):
    def __init__(self, step_size: float, *args: Any, **kwargs: Any) -> None:
        """
        Pretrain Dataset Base Class.

        Args:
            step_size (float): Step size for the dataset.
        """
        super().__init__()
        self.step_size = self.check_step_size(step_size)

    def check_step_size(self, step_size: float) -> float:
        """
        Check if the step size is valid.

        Args:
            step_size (float): Step size to check.

        Raises:
            ValueError: Raised if the step size is non-positive.

        Returns:
            float: Validated step size.
        """
        if step_size <= 0:
            raise ValueError("Step size must be positive.")
        return step_size

    def check_num_data(self, num_data: int) -> int:
        """
        Check if the number of samples is valid.

        Args:
            num_data (int): Number of samples to check.

        Raises:
            ValueError: Raised if the number of samples is non-positive.

        Returns:
            int: Validated number of samples.
        """
        if num_data <= 0 or not isinstance(num_data, int):
            raise ValueError("Number of samples must be a positive integer.")
        return num_data

    def __len__(self) -> int:
        """
        Returns the number of samples.

        Returns:
            int: Number of samples.
        """
        return self.num_data

    def __getitem__(
        self, index: int
    ) -> Tuple[Float[torch.Tensor, "num dim"], Float[torch.Tensor, "num dim"]]:
        """
        Returns the state and action at the given index.

        Returns:
            Tuple[Float[torch.Tensor, "num dim"], Float[torch.Tensor, "num dim"]]: State and action.
        """
        return self.mock_state[index].double(), self.mock_step_size[index].double()

    def inverse_softplus(
        self, /, x: Float[torch.Tensor, "num dim"]
    ) -> Float[torch.Tensor, "num dim"]:
        """
        Inverse softplus function.

        Returns:
            Float[torch.Tensor, "num dim"]: Output.
        """
        return x + torch.log1p(-torch.exp(-x))

    @abstractmethod
    def mock_state_generator(
        self, *args: Any, **kwargs: Any
    ) -> Float[torch.Tensor, "num dim"]:
        """
        Mock state generator.

        Returns:
            Float[torch.Tensor, "num dim"]: State.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def mock_step_size_generator(self) -> Float[torch.Tensor, "num dim"]:
        """
        Mock step size generator.

        Returns:
            Float[torch.Tensor, "num dim"]: Step size.
        """
        return self.inverse_softplus(
            torch.repeat_interleave(
                torch.tensor([self.step_size]), self.num_data << 1
            ).view(self.num_data, -1)
        )


class PretrainMockDataset(PretrainDatasetBase):
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
        initial_sample: Float[torch.Tensor, "num dim"],
        step_size: float = 1.0,
        num_data: int = 1_000,
        mag: float = 10.0,
    ) -> None:
        """
        Pretrain Mock Dataset for Actor.

        Args:
            initial_sample (Float[torch.Tensor, "num dim"]): Initial sample.
            step_size (float, optional): Step size. Defaults to 1.0.
            num_data (int, optional): Number of samples. Defaults to 1_000.
            mag (float, optional): Magnification. Defaults to 10.0.

        Raises:
            ValueError: If step size is non-positive.
            ValueError: If number of samples is non-positive.
            ValueError: If magnification is non-positive.
        """
        super().__init__(step_size)
        self.mag = self.check_mag(mag)

        self.initial_sample = initial_sample
        self.sample_dim = len(initial_sample)

        self.num_data = self.check_num_data(num_data)

        self.mock_state = self.mock_state_generator(num_data)
        self.mock_step_size = self.mock_step_size_generator()

    def check_mag(self, mag: float) -> float:
        """
        Check if the magnification is valid.

        Args:
            mag (float): Magnification to check.

        Raises:
            ValueError: Raised if the magnification is non-positive.

        Returns:
            float: Validated magnification.
        """
        if mag <= 0:
            raise ValueError("Magnification must be positive.")
        return mag

    def mock_state_generator(self, num_data: int) -> Float[torch.Tensor, "num dim"]:
        """
        Mock state generator.

        Args:
            num_data (int): Number of samples.

        Returns:
            Float[torch.Tensor, "num dim"]: State.
        """
        multivariate_normal = torch.distributions.MultivariateNormal(
            loc=self.initial_sample,
            covariance_matrix=(
                self.mag * torch.eye(self.sample_dim, dtype=torch.float64)
            ),
        )

        return multivariate_normal.sample((num_data, self.sample_dim)).view(
            -1, self.sample_dim << 1
        )


class PretrainPosteriorDBDataset(PretrainDatasetBase):
    def __init__(
        self,
        model_name: str,
        posteriordb_path: str,
        step_size: float = 1.0,
    ) -> None:
        super().__init__(step_size)

        self.mock_state = self.mock_state_generator(model_name, posteriordb_path)
        self.num_data = self.check_num_data(self.mock_state.shape[0])
        self.mock_step_size = self.mock_step_size_generator()

    def mock_state_generator(
        self, model_name: str, posteriordb_path: str
    ) -> Float[torch.Tensor, "num dim"]:
        """
        Mock state generator.

        Args:
            model_name (str): Model name.
            posteriordb_path (str): PosteriorDB path.

        Returns:
            Float[torch.Tensor, "num dim"]: State.
        """
        gold_standard = Toolbox.gold_standard(model_name, posteriordb_path)
        gold_standard_torch = torch.from_numpy(gold_standard).double()
        mock_state = gold_standard_torch.tile((2,))
        return mock_state


class PretrainFactory:
    @staticmethod
    def train(
        actor: PolicyNetwork,
        mock_dataset: PretrainDatasetBase,
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
            initial_sample (Float[torch.Tensor, "num dim"]): Initial sample.
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
        # mock_dataset = PretrainMockDataset(initial_sample, step_size, num_data, mag)
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
