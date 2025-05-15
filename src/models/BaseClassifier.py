import torch

from abc import ABC, abstractmethod
from typing import Any


class BaseClassifier(ABC):
    """
    Abstract base class for classifier models, defining the necessary methods
    for training and validation workflows.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor (e.g. class logits).
        """
        pass

    @abstractmethod
    def training_step(self, batch: Any) -> torch.Tensor:
        """
        Computes the training loss for a single batch.

        Args:
            batch (Any): A batch of training data, typically a tuple (inputs, targets).

        Returns:
            torch.Tensor: Computed training loss.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any) -> dict[str, float]:
        """
        Computes validation metrics for a single batch.

        Args:
            batch (Any): A batch of validation data, typically a tuple (inputs, targets).

        Returns:
            Dict[str, float]: Dictionary with validation loss and accuracy.
        """
        pass

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Sets up the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer instance used in training.
        """
        pass
