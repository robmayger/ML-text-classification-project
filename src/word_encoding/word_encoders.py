from abc import ABC, abstractmethod
from collections import Counter


class EncoderInterface(ABC):

    @abstractmethod
    def encode(self, tokens: list[str], padding: bool = True, max_len: int = 100) -> list[int]:
        ...


class BasicEncoder(EncoderInterface):

    def __init__(self, tokens: list[str]):
        self._vocab = self._build_vocab(tokens)

    def _build_vocab(self, tokens: list[str], min_freq: int = 2) -> str:
        counter = Counter()
        for token in tokens:
            counter.update(tokens)
        
        self._vocab = {word: idx + 2 for idx, (word, count) in enumerate(self._counter.items()) if count >= min_freq}
        self._vocab['<PAD>'] = 0
        self._vocab['<UNK>'] = 1

    def encode(self, tokens: list[str], max_len: int = None) -> list[int]:
        if max_len is not None and max_len < 1:
            raise ValueError("`max_len` must be a positive integer (>= 1).")

        encoding = [self._vocab.get(token, self._vocab['<UNK>']) for token in tokens]

        if max_len is not None:
            encoding = self._pad_sequence(encoding, max_len, self._vocab['<PAD>'])

        return encoding

    @staticmethod
    def _pad_sequence(seq: list[int], max_len: int, pad_value: int) -> list[int]:
        if len(seq) < max_len:
            return seq + [pad_value] * (max_len - len(seq))
        else:
            return seq[:max_len]
