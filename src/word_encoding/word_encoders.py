from abc import ABC, abstractmethod
from collections import Counter
import numpy as np


class EncoderInterface(ABC):

    @abstractmethod
    def encode(self, tokens: list[str], padding: bool = True, max_len: int = 100) -> np.ndarray:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass


class BasicEncoder(EncoderInterface):

    def __init__(self, tokens: list[str]):
        self._vocab = self._build_vocab(tokens)

    def encode(self, tokens: list[str], max_len: int = None) -> list[int]:
        if max_len is not None and max_len < 1:
            raise ValueError("`max_len` must be a positive integer (>= 1).")

        encoding = np.array([self._vocab.get(token, self._vocab['<UNK>']) for token in tokens])

        if max_len is not None:
            encoding = self._pad_sequence(encoding, max_len, self._vocab['<PAD>'])

        return encoding
    
    def get_vocab_size(self) -> int:
        return len(self._vocab)
    
    @staticmethod
    def _build_vocab(tokens: list[str], min_freq: int = 2) -> dict[str, int]:
        counter = Counter(tokens)
        filtered_counter = Counter({k: v for k, v in counter.items() if v >= min_freq})
        vocab = {word: idx + 2 for idx, (word, count) in enumerate(filtered_counter.items())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    @staticmethod
    def _pad_sequence(seq: np.ndarray, max_len: int, pad_value: int) -> np.ndarray:
        if len(seq) < max_len:
            pad_width = max_len - len(seq)
            return np.concatenate([seq, np.full(pad_width, pad_value, dtype=seq.dtype)])
        else:
            return seq[:max_len]
