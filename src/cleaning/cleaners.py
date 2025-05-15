from abc import ABC, abstractmethod
import re


class TextCleanerInterface(ABC):
    """
    Abstract base class that defines a text cleaning interface.
    """
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Cleans a single input text string.

        Args:
            text (str): The raw input text.

        Returns:
            str: The cleaned text.
        """
        ...


class BasicTextCleaner(TextCleanerInterface):
    """
    A basic implementation of the TextCleanerInterface that:
    - Converts text to lowercase
    - Removes HTML tags
    - Removes non-alphabetic characters
    - Normalizes whitespace
    """
    def __init__(self):
        """
        Initializes the BasicTextCleaner.
        """
        super().__init__()

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text using basic text normalization rules.

        Args:
            text (str): The raw input text.

        Returns:
            str: The cleaned and normalized text.
        """
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
