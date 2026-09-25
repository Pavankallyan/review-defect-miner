"""End-to-end pipeline: generate -> preprocess -> train -> topics -> prioritize.

Run from the project root:  python run.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import generate_reviews
import prioritize
import topics
import train


def main() -> None:
    print("=" * 60)
    print("STEP 1/4: generating synthetic review dataset")
    print("=" * 60)
    generate_reviews.main()

    print("\n" + "=" * 60)
    print("STEP 2/4: training classifiers")
    print("=" * 60)
    metrics = train.main()

    print("\n" + "=" * 60)
    print("STEP 3/4: topic modeling on defect reviews")
    print("=" * 60)
    topics.main()

    print("\n" + "=" * 60)
    print("STEP 4/4: building QA triage queue")
    print("=" * 60)
    prioritize.main()

    print("\nPipeline complete.")
    print("Binary defect classifier :", metrics["binary"])
    print("Category classifier     :", metrics["category"])
    print("\nLaunch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
