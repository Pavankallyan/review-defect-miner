"""Topic modeling on the defect reviews with sklearn LDA.

Uses a CountVectorizer (LDA needs raw counts, not TF-IDF) and fits
LatentDirichletAllocation with 6 topics over defect-only reviews.

Run:  python src/topics.py   (from the project root)
Prints the top keywords per topic and saves them to models/topics.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from preprocess import STOPWORDS, clean_text

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 6
N_TOP_WORDS = 10
SEED = 42

# Device/product filler words that would otherwise dominate every topic.
TOPIC_STOPWORDS = set(STOPWORDS) | {
    "thermostat", "doorbell", "camera", "lock", "bulb", "plug", "speaker",
    "sensor", "detector", "video", "living", "room", "kitchen", "bedroom",
    "garage", "smoke", "smart", "nestmate", "device", "devices",
}


def fit_topics(n_topics: int = N_TOPICS, seed: int = SEED) -> dict:
    df = pd.read_csv(DATA_PATH)
    defects = df[df["is_defect"] == 1]["text"]

    count_vec = CountVectorizer(
        stop_words=list(TOPIC_STOPWORDS),
        ngram_range=(1, 1),
        min_df=5,
        max_features=4000,
    )
    X_counts = count_vec.fit_transform(defects.apply(clean_text))

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=seed, learning_method="batch",
        max_iter=25,
    )
    lda.fit(X_counts)

    vocab = count_vec.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:N_TOP_WORDS]
        topics[f"topic_{i}"] = [vocab[j] for j in top_idx]
    return topics


def main() -> dict:
    topics = fit_topics()
    for name, words in topics.items():
        print(f"{name}: {', '.join(words)}")
    with open(MODELS_DIR / "topics.json", "w") as f:
        json.dump(topics, f, indent=2)
    print(f"Saved topics to {MODELS_DIR / 'topics.json'}")
    return topics


if __name__ == "__main__":
    main()
