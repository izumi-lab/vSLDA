"""Download required NLTK data for this project."""

import nltk


def _wordnet_is_installed() -> bool:
    try:
        nltk.data.find("corpora/wordnet.zip")
    except LookupError:
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            return False
    return True


def main() -> None:
    """Download the WordNet corpus used by English lemmatization."""
    if _wordnet_is_installed():
        print("NLTK wordnet corpus is already installed.")
        return

    print("Downloading NLTK wordnet corpus...")
    if not nltk.download("wordnet"):
        raise RuntimeError("Failed to download NLTK wordnet corpus.")
    print("Done.")


if __name__ == "__main__":
    main()
