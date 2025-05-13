import torch

from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    @abstractmethod
    def configure_optimizers(self):
        pass
