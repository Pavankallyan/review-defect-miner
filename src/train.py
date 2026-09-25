"""Train the two classifiers on the generated review dataset.

Model A (binary):    defect (1) vs non-defect (0) -- LogisticRegression on TF-IDF
Model B (multiclass): defect category among the 6 defect classes -- LinearSVC on TF-IDF

Run:  python src/train.py   (from the project root)
Saves: models/vectorizer.joblib, models/binary_clf.joblib, models/category_clf.joblib
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from preprocess import clean_text, make_vectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFECT_CATEGORIES = [
    "crash", "battery_drain", "connectivity",
    "ui_bug", "performance", "login_auth",
]

TEST_SIZE = 0.25
SEED = 42


def evaluate(name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"{name}: accuracy={acc:.4f}  weighted-F1={f1:.4f}")
    return {"accuracy": round(acc, 4), "f1_weighted": round(f1, 4)}


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} reviews from {DATA_PATH}")

    vectorizer = make_vectorizer()
    clean = df["text"].apply(clean_text)
    X = vectorizer.fit_transform(clean)

    # ---- Model A: binary defect classifier ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, df["is_defect"], test_size=TEST_SIZE, random_state=SEED,
        stratify=df["is_defect"],
    )
    binary_clf = LogisticRegression(max_iter=1000, C=2.0, random_state=SEED)
    binary_clf.fit(X_train, y_train)
    metrics_a = evaluate("Binary defect classifier (test)", y_test, binary_clf.predict(X_test))

    # ---- Model B: defect-category classifier (defects only) ----
    defects = df[df["is_defect"] == 1].reset_index(drop=True)
    Xd = vectorizer.transform(defects["text"].apply(clean_text))
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        Xd, defects["category"], test_size=TEST_SIZE, random_state=SEED,
        stratify=defects["category"],
    )
    category_clf = LinearSVC(C=1.0, random_state=SEED)
    category_clf.fit(Xd_train, yd_train)
    metrics_b = evaluate("Category classifier (test)", yd_test, category_clf.predict(Xd_test))

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    joblib.dump(binary_clf, MODELS_DIR / "binary_clf.joblib")
    joblib.dump(category_clf, MODELS_DIR / "category_clf.joblib")
    print(f"Saved models to {MODELS_DIR}")

    return {"binary": metrics_a, "category": metrics_b}


if __name__ == "__main__":
    main()
