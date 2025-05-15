from abc import ABC, abstractmethod


class TokeniserInterface(ABC):
    """
    Abstract base class for a tokeniser.
    """
    @abstractmethod
    def tokenise(self, text: str) -> str:
        """
        Tokenises the input text into a list of tokens.

        Args:
            text (str): The raw input text.

        Returns:
            List[str]: A list of tokens.
        """
        pass


class BasicTokeniser(TokeniserInterface):
    """
    A simple whitespace tokeniser that truncates tokens to a maximum length.

    Attributes:
        max_len (int): Maximum number of tokens to return.
    """
    def __init__(self, max_len: int = 200) -> None:
        """
        Initializes the BasicTokeniser with an optional max length.

        Args:
            max_len (int): Maximum number of tokens to retain. Defaults to 200.
        """
        super().__init__()
        self.max_len = max_len

    def tokenise(self, text: str) -> list[str]:
        """
        Tokenises the input text by whitespace and truncates to max_len.

        Args:
            text (str): The raw input text.

        Returns:
            List[str]: A list of tokens.
        """
        return text.split()[:self.max_len]
