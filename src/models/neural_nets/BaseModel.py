import torch.nn as nn

from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def forward(self, x):
        pass


class PytorchBaseModel(BaseModel, nn.Module):
    pass
