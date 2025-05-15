import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for models, enforcing the implementation of the forward method.
    """
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass


class PytorchBaseModel(BaseModel, nn.Module):
    """
    A base class combining an abstract model interface with PyTorch's nn.Module.
    This allows consistent typing and integration with PyTorch's training utilities.
    """
    pass

