"""Unit tests for src/prioritize.py — the QA triage-queue scoring module."""

import math

import numpy as np
import pandas as pd
import pytest

from prioritize import (
    RECENCY_HALFLIFE_DAYS,
    SEVERITY_WEIGHTS,
    build_queue,
    recency_factor,
)


def make_reviews(**overrides):
    rows = [
        {"review_id": 1, "text": "app crashes on launch",
         "category": "crash", "star_rating": 1, "date": "2024-01-10",
         "is_defect": 1, "app_version": "3.1.0"},
        {"review_id": 2, "text": "another crash, phone freezes",
         "category": "crash", "star_rating": 2, "date": "2024-01-05",
         "is_defect": 1, "app_version": "3.1.0"},
        {"review_id": 3, "text": "button is slightly misaligned",
         "category": "ui_bug", "star_rating": 4, "date": "2024-01-10",
         "is_defect": 1, "app_version": "3.2.0"},
        {"review_id": 4, "text": "great app, love it",
         "category": "ui_bug", "star_rating": 5, "date": "2024-01-10",
         "is_defect": 0, "app_version": "3.2.0"},
    ]
    rows[0].update(overrides.pop("row0", {}))
    return pd.DataFrame(rows).assign(**overrides)


def test_recency_factor_newest_is_one():
    dates = pd.Series(["2024-01-05", "2024-01-10"])
    factors = recency_factor(dates)
    assert factors.iloc[-1] == pytest.approx(1.0)


def test_recency_factor_half_life():
    # One half-life (90 days) old should score ~0.5 against an explicit reference.
    dates = pd.Series(["2024-01-10"])
    factors = recency_factor(dates, reference="2024-04-09")
    days = 90.0
    expected = math.exp(-math.log(2) * days / RECENCY_HALFLIFE_DAYS)
    assert factors.iloc[0] == pytest.approx(expected)


def test_recency_factor_older_scores_lower():
    dates = pd.Series(["2024-01-01", "2024-01-10"])
    factors = recency_factor(dates)
    assert factors.iloc[0] < factors.iloc[1]


def test_build_queue_drops_non_defects():
    queue = build_queue(make_reviews())
    assert set(queue["is_defect"]) == {1}
    assert 4 not in queue["review_id"].values  # the non-defect review


def test_build_queue_empty_defects():
    df = make_reviews().assign(is_defect=0)
    queue = build_queue(df)
    assert queue.empty


def test_build_queue_rank_is_sorted_by_score():
    queue = build_queue(make_reviews())
    scores = queue["triage_score"].to_numpy()
    assert np.all(scores[:-1] >= scores[1:])
    assert list(queue["rank"]) == sorted(queue["rank"])
    assert queue["rank"].iloc[0] == 1


def test_severity_weighting_moves_crashes_above_ui_bugs():
    queue = build_queue(make_reviews())
    top = queue.iloc[0]
    # crash severity (5.0) dominates ui_bug (2.0) on otherwise identical rows
    assert top["category"] == "crash"
    assert top["review_id"] == 1  # fresher + angrier crash ranks first


def test_rating_factor_prefers_angrier_users():
    # Two crashes with identical everything except star rating.
    df = make_reviews(
        row0={"review_id": 10, "category": "crash", "star_rating": 5,
              "date": "2024-01-10", "is_defect": 1},
    )
    queue = build_queue(df)
    q1 = queue.set_index("review_id").loc
    assert q1[10, "triage_score"] < q1[2, "triage_score"]


def test_unknown_category_falls_back_to_weight_one():
    df = make_reviews(row0={"review_id": 11, "category": "mystery_bug"})
    queue = build_queue(df)
    weight = queue.set_index("review_id").loc[11, "severity_weight"]
    assert weight == pytest.approx(1.0)


def test_severity_weights_match_known_classes():
    assert SEVERITY_WEIGHTS["crash"] == 5.0
    assert SEVERITY_WEIGHTS["ui_bug"] == 2.0
    assert SEVERITY_WEIGHTS["crash"] > SEVERITY_WEIGHTS["login_auth"]


def test_score_is_positive_and_finite():
    queue = build_queue(make_reviews())
    assert (queue["triage_score"] > 0).all()
    assert np.isfinite(queue["triage_score"]).all()
