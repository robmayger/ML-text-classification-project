from abc import ABC, abstractmethod
from collections import Counter
import numpy as np


class EncoderInterface(ABC):
    """
    Abstract base class for text encoders.
    """
    @abstractmethod
    def encode(self, tokens: list[str], padding: bool = True, max_len: int = 100) -> np.ndarray:
        """
        Encodes a list of tokens into a sequence of integers.

        Args:
            tokens (List[str]): List of string tokens to encode.
            padding (bool): Whether to pad the sequence to `max_len`. Defaults to True.
            max_len (int): The maximum sequence length. Defaults to 100.

        Returns:
            np.ndarray: Encoded and optionally padded sequence as a NumPy array.
        """
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Returns the number of unique tokens in the vocabulary.

        Returns:
            int: Vocabulary size.
        """
        pass


class BasicEncoder(EncoderInterface):
    """
    A basic encoder that builds a vocabulary from tokens and encodes new inputs to integer sequences.
    Applies padding and handles unknown words with special tokens.
    """
    def __init__(self, tokens: list[str]) -> None:
        """
        Initializes the encoder and builds a vocabulary.

        Args:
            tokens (List[str]): A list of tokens to build the vocabulary from.
        """
        self._vocab = self._build_vocab(tokens)

    def encode(self, tokens: list[str], max_len: int = None) -> list[int]:
        """
        Encodes the given tokens into a sequence of indices based on the internal vocabulary.

        Args:
            tokens (List[str]): Tokens to encode.
            max_len (int, optional): Desired sequence length. If specified,
                                     the sequence is padded or truncated accordingly.

        Returns:
            np.ndarray: Encoded sequence of token indices.
        """
        if max_len is not None and max_len < 1:
            raise ValueError("`max_len` must be a positive integer (>= 1).")

        encoding = np.array([self._vocab.get(token, self._vocab['<UNK>']) for token in tokens])

        if max_len is not None:
            encoding = self._pad_sequence(encoding, max_len, self._vocab['<PAD>'])

        return encoding
    
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        Returns:
            int: Vocabulary size.
        """
        return len(self._vocab)
    
    @staticmethod
    def _build_vocab(tokens: list[str], min_freq: int = 2) -> dict[str, int]:
        """
        Builds a vocabulary from tokens, keeping only those that appear at least `min_freq` times.

        Args:
            tokens (List[str]): List of tokens.
            min_freq (int): Minimum frequency threshold for inclusion in the vocabulary.

        Returns:
            Dict[str, int]: Mapping from token to index.
        """
        counter = Counter(tokens)
        filtered_counter = Counter({k: v for k, v in counter.items() if v >= min_freq})
        vocab = {word: idx + 2 for idx, (word, count) in enumerate(filtered_counter.items())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    @staticmethod
    def _pad_sequence(seq: np.ndarray, max_len: int, pad_value: int) -> np.ndarray:
        """
        Pads or truncates a sequence to a fixed length.

        Args:
            seq (np.ndarray): The sequence to pad or truncate.
            max_len (int): The desired length of the output sequence.
            pad_value (int): The value to use for padding.

        Returns:
            np.ndarray: The padded or truncated sequence.
        """
        if len(seq) < max_len:
            pad_width = max_len - len(seq)
            return np.concatenate([seq, np.full(pad_width, pad_value, dtype=seq.dtype)])
        else:
            return seq[:max_len]
