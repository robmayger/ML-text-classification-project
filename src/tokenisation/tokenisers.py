from abc import ABC, abstractmethod


class TokeniserInterface(ABC):

    @abstractmethod
    def tokenise(self, text: str) -> str:
        ...


class BasicTokeniser(TokeniserInterface):

    def __init__(self):
        super().__init__()

    def tokenise(self, text: str) -> str:
        return text.split()
