"""QA triage queue: score and rank defect reviews for fixing.

Score formula (per defect review):
    score = severity_weight(category)
          x frequency_factor(category)      -- how common this defect class is
          x recency_factor(date)            -- fresher reports matter more
          x (6 - star_rating)               -- angrier users rank higher

All factors are normalized to roughly [0, 1] except the weights, so the
final score is a relative ordering, not an absolute number.

Run:  python src/prioritize.py   (from the project root)
Writes: data/triage_queue.csv  (all defect reviews, ranked by score)
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "reviews.csv"
OUT_PATH = ROOT / "data" / "triage_queue.csv"

# QA-style severity: how bad is this class of defect for the user?
SEVERITY_WEIGHTS = {
    "crash": 5.0,         # app unusable
    "login_auth": 4.5,    # locked out entirely
    "connectivity": 4.0,  # device unreachable
    "battery_drain": 3.0, # degrades phone experience
    "performance": 3.0,   # slow / unresponsive
    "ui_bug": 2.0,        # cosmetic / confusing
}

RECENCY_HALFLIFE_DAYS = 90.0


def recency_factor(dates: pd.Series, reference: str | None = None) -> pd.Series:
    """Exponential decay: 1.0 for the newest review, ~0.5 after one half-life."""
    dates = pd.to_datetime(dates)
    ref = pd.to_datetime(reference) if reference else dates.max()
    days_old = (ref - dates).dt.total_seconds() / 86400.0
    return np.exp(-np.log(2) * days_old / RECENCY_HALFLIFE_DAYS)


def build_queue(df: pd.DataFrame | None = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(DATA_PATH)
    defects = df[df["is_defect"] == 1].copy()
    if defects.empty:
        return defects

    # frequency factor: share of all defects that share this category
    freq = defects["category"].value_counts(normalize=True)
    defects["frequency_factor"] = defects["category"].map(freq)

    defects["severity_weight"] = defects["category"].map(SEVERITY_WEIGHTS).fillna(1.0)
    defects["recency_factor"] = recency_factor(defects["date"])
    defects["rating_factor"] = 6 - defects["star_rating"]

    defects["triage_score"] = (
        defects["severity_weight"]
        * defects["frequency_factor"]
        * defects["recency_factor"]
        * defects["rating_factor"]
    )
    queue = defects.sort_values("triage_score", ascending=False).reset_index(drop=True)
    queue["rank"] = queue.index + 1
    return queue


def main() -> pd.DataFrame:
    queue = build_queue()
    cols = ["rank", "review_id", "triage_score", "severity_weight", "category",
            "star_rating", "date", "app_version", "text"]
    queue[cols].to_csv(OUT_PATH, index=False)
    print(f"Wrote ranked queue of {len(queue)} defects to {OUT_PATH}")
    print("\nTop 10 by triage score:")
    print(queue[["rank", "triage_score", "category", "star_rating", "date"]].head(10)
          .to_string(index=False))
    print("\nQueue by category:")
    print(queue["category"].value_counts().to_string())
    return queue


if __name__ == "__main__":
    main()
