from abc import ABC, abstractmethod


class TokeniserInterface(ABC):

    @abstractmethod
    def tokenise(self, text: str) -> str:
        ...


class BasicTokeniser(TokeniserInterface):

    def __init__(self, max_len: int = 200):
        super().__init__()
        self.max_len = max_len

    def tokenise(self, text: str) -> str:
        return text.split()[:self.max_len]
