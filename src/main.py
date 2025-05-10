from sklearn.datasets import fetch_20newsgroups
from cleaning.cleaners import BasicTextCleaner
from tokenisation.tokenisers import BasicTokeniser
from word_encoding.word_encoders import BasicEncoder

from itertools import chain


if __name__ == "__main__":
    data = fetch_20newsgroups(subset='train')9

    X = data.data
    y = data.target

    # Optional: Get the category names
    label_names = data.target_names

    cleaner = BasicTextCleaner()
    X_clean = [cleaner.clean_text(text) for text in X]

    tokeniser = BasicTokeniser()
    X_tokens = [tokeniser.tokenise(text) for text in X_clean]

    max_doc_len = max(map(lambda x: len(x), X_tokens))

    flat = list(chain.from_iterable(X_tokens))
    encoder = BasicEncoder(flat)
    X_encodings = [encoder.encode(doc, max_len=max_doc_len) for doc in X_tokens]
    print(X_encodings)
