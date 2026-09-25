"""Text cleaning + TF-IDF vectorization shared by training and inference."""

import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "with", "is", "it", "this", "that", "these",
    "those", "i", "my", "me", "we", "you", "your", "they", "their", "he",
    "she", "his", "her", "its", "as", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "not", "no", "so", "such", "too", "very",
    "just", "from", "up", "out", "about", "into", "over", "after", "before",
    "when", "while", "how", "what", "which", "who", "whom", "there", "here",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "only", "own", "same", "than", "now", "app",
}


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/emails/noise, keep letters and spaces only."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_vectorizer() -> TfidfVectorizer:
    """Plain TF-IDF vectorizer (no custom callables, so it pickles standalone).

    Clean text with `clean_text()` BEFORE vectorizing, e.g.:
        texts = [clean_text(t) for t in raw_texts]
        X = vectorizer.fit_transform(texts)
    """
    return TfidfVectorizer(
        stop_words=list(STOPWORDS),
        ngram_range=(1, 2),
        min_df=3,
        max_features=6000,
    )


def load_vectorizer():
    """Load the fitted TF-IDF vectorizer saved by src/train.py."""
    import joblib

    path = MODELS_DIR / "vectorizer.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {path}. Run `python src/train.py` first."
        )
    return joblib.load(path)


def vectorize_texts(vectorizer, texts):
    """Clean raw texts, then transform with a fitted vectorizer."""
    return vectorizer.transform([clean_text(t) for t in texts])
