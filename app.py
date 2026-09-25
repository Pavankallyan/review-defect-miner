"""Streamlit triage dashboard for the review-defect-miner project.

Run:  streamlit run app.py   (from the project root)

Layout:
  - Sidebar: category filter + minimum severity slider
  - Main: ranked defect queue table, category distribution chart,
          LDA topic keywords, and a sample review drill-down.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
REVIEWS_PATH = ROOT / "data" / "reviews.csv"
QUEUE_PATH = ROOT / "data" / "triage_queue.csv"
TOPICS_PATH = ROOT / "models" / "topics.json"

SEVERITY_WEIGHTS = {
    "crash": 5.0, "login_auth": 4.5, "connectivity": 4.0,
    "battery_drain": 3.0, "performance": 3.0, "ui_bug": 2.0,
}


# ---------------- data loading (pure functions, testable headlessly) ----------------

def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_queue(path: Path = QUEUE_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_topics(path: Path = TOPICS_PATH) -> dict:
    import json

    return json.loads(path.read_text())


def category_counts(queue: pd.DataFrame) -> pd.Series:
    return queue["category"].value_counts()


def filter_queue(queue: pd.DataFrame, categories: list, min_severity: float) -> pd.DataFrame:
    mask = queue["category"].isin(categories) & (queue["severity_weight"] >= min_severity)
    return queue[mask].reset_index(drop=True)


def queue_summary(queue: pd.DataFrame) -> dict:
    return {
        "n_defects": int(len(queue)),
        "mean_score": float(queue["triage_score"].mean()) if len(queue) else 0.0,
        "top_category": str(queue["category"].mode().iat[0]) if len(queue) else "",
        "categories": sorted(queue["category"].unique().tolist()),
    }


# ---------------- UI ----------------

def main() -> None:
    st.set_page_config(page_title="NestMate Review Triage", layout="wide")
    st.title("Review Defect Miner -- QA Triage Queue")
    st.caption(
        "NLP pipeline mining synthetic app reviews (fictional NestMate smart-home app) "
        "for software defects, ranked like a QA triage queue."
    )

    queue = load_queue()
    topics = load_topics() if TOPICS_PATH.exists() else {}

    with st.sidebar:
        st.header("Filters")
        all_cats = sorted(queue["category"].unique().tolist())
        selected = st.multiselect("Defect category", all_cats, default=all_cats)
        min_sev = st.slider("Minimum severity weight", 1.0, 5.0, 1.0, step=0.5)

    filtered = filter_queue(queue, selected, min_sev)
    summary = queue_summary(filtered)

    c1, c2, c3 = st.columns(3)
    c1.metric("Defects in queue", summary["n_defects"])
    c2.metric("Mean triage score", f"{summary['mean_score']:.3f}")
    c3.metric("Top category", summary["top_category"])

    st.subheader("Ranked defect queue")
    st.dataframe(
        filtered[["rank", "triage_score", "severity_weight", "category",
                  "star_rating", "date", "app_version", "text"]],
        use_container_width=True, height=400,
    )

    st.subheader("Defect category distribution")
    counts = category_counts(filtered)
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot.bar(ax=ax, color="#4e79a7")
    ax.set_ylabel("Number of defect reviews")
    ax.set_xlabel("Category")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Discovered topics (LDA over defect reviews)")
    for name, words in topics.items():
        st.markdown(f"**{name}**: {', '.join(words)}")

    st.subheader("Review drill-down")
    if len(filtered):
        review_id = st.selectbox(
            "Pick a review", filtered["review_id"].tolist(),
            format_func=lambda rid: f"{rid} -- {filtered.set_index('review_id').loc[rid, 'category']}",
        )
        row = filtered.set_index("review_id").loc[review_id]
        st.markdown(f"**{review_id}** | {row['category']} | "
                    f"stars: {row['star_rating']} | {row['date']} | v{row['app_version']}")
        st.write(row["text"])
        st.progress(min(1.0, row["triage_score"] / filtered["triage_score"].max()))
        st.caption(f"Triage score: {row['triage_score']:.4f}")
    else:
        st.info("No defects match the current filters.")


if __name__ == "__main__":
    main()
