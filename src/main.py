from sklearn.datasets import fetch_20newsgroups
from text_cleaning import BasicTextCleaner


if __name__ == "__main__":
    data = fetch_20newsgroups(subset='train')

    X = data.data
    y = data.target

    # Optional: Get the category names
    label_names = data.target_names

    cleaner = BasicTextCleaner()

    X_clean = [cleaner.clean_text(text) for text in X]


