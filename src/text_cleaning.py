from abc import ABC, abstractmethod
import re


class TextCleaner(ABC):

    @abstractmethod
    def clean_text(self, text: str) -> str:
        pass


class BasicTextCleaner(TextCleaner):

    def __init__(self):
        super().__init__()

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
