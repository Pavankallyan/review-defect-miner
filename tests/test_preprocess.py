"""Unit tests for src/preprocess.py (text cleaning + TF-IDF vectorization)."""

import pytest

from preprocess import STOPWORDS, clean_text, load_vectorizer, make_vectorizer, vectorize_texts


# A corpus where each content word appears in at least 3 documents
# (the vectorizer uses min_df=3), and "app" appears in 4 documents
# (so the stopword assertion is meaningful).
CORPUS = [
    "The app crashes on the login screen every time",
    "App freezes when I try to upload a photo",
    "Login screen fails and the app crashes again",
    "App crashes after the latest update, please fix",
    "Cannot login, button does nothing on click",
    "Login screen is blank, cannot sign in",
]


def test_clean_text_lowercases():
    assert clean_text("APP Crashes on LOGIN") == "app crashes on login"


def test_clean_text_strips_urls():
    assert clean_text("see https://example.com/help for more info") == "see for more info"


def test_clean_text_strips_www_urls():
    assert clean_text("go to www.example.com/page now") == "go to now"


def test_clean_text_strips_emails():
    assert clean_text("contact support@example.com please") == "contact please"


def test_clean_text_drops_digits_and_punctuation():
    assert clean_text("  Error 500!!  crashed 3 times...  ") == "error crashed times"


def test_clean_text_empty_and_whitespace():
    assert clean_text("") == ""
    assert clean_text("   \t\n  ") == ""


def test_clean_text_collapses_whitespace():
    assert clean_text("login   button\n\tdoes\tnothing") == "login button does nothing"


def test_clean_text_non_string_input():
    assert clean_text(12345) == ""
    assert clean_text(None) == "none"


def test_vectorizer_shape_and_nonzero():
    v = make_vectorizer()
    X = v.fit_transform([clean_text(t) for t in CORPUS])
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] == len(v.vocabulary_)
    assert X.nnz > 0


def test_vectorizer_min_df_filters_rare_words():
    v = make_vectorizer()
    v.fit([clean_text(t) for t in CORPUS])
    # "photo" appears in only one document -> dropped by min_df=3
    assert "photo" not in v.vocabulary_
    # "crashes" appears in 3 documents -> kept
    assert "crashes" in v.vocabulary_


def test_vectorizer_drops_domain_stopword_app():
    assert "app" in STOPWORDS
    v = make_vectorizer()
    v.fit([clean_text(t) for t in CORPUS])
    assert "app" not in v.vocabulary_


def test_vectorizer_has_bigram_features():
    v = make_vectorizer()
    v.fit([clean_text(t) for t in CORPUS])
    bigrams = [f for f in v.get_feature_names_out() if " " in f]
    assert len(bigrams) > 0


def test_vectorizer_max_features_cap():
    v = make_vectorizer()
    assert v.max_features == 6000
    assert v.ngram_range == (1, 2)


def test_vectorize_texts_cleans_before_transform():
    v = make_vectorizer()
    v.fit([clean_text(t) for t in CORPUS])
    noisy = ["The app CRASHES on login!! See https://x.com/help"]
    cleaned = [clean_text(noisy[0])]
    a = vectorize_texts(v, noisy)
    b = vectorize_texts(v, cleaned)
    assert a.shape == b.shape
    assert (a != b).nnz == 0


def test_load_vectorizer_missing_file_raises_helpful_error(tmp_path, monkeypatch):
    import preprocess

    monkeypatch.setattr(preprocess, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Run `python src/train.py` first"):
        load_vectorizer()
