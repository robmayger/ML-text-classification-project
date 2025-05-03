from abc import ABC, abstractmethod
import re


class TextCleanerInterface(ABC):

    @abstractmethod
    def clean_text(self, text: str) -> str:
        ...


class BasicTextCleaner(TextCleanerInterface):

    def __init__(self):
        super().__init__()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
