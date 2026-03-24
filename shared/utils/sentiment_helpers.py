"""
sentiment_helpers.py
--------------------
Shared helpers for VADER-based sentiment scoring and classification.

Provides:
  - score_text()        : score a single cleaned string
  - classify_sentiment(): map compound score to label
  - score_dataframe()   : apply scoring to a full DataFrame column

Usage:
    from shared.utils.sentiment_helpers import score_dataframe
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# ── Initialise VADER once (expensive to reload repeatedly) ────────────────────
_analyzer = SentimentIntensityAnalyzer()


# ── Thresholds (VADER standard recommendation) ────────────────────────────────
POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05


def score_text(text: str) -> dict:
    """
    Score a single text string using VADER.

    Returns a dict with keys:
        neg, neu, pos   : proportion scores (sum to 1.0)
        compound        : normalised aggregate score in [-1, 1]
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}
    return _analyzer.polarity_scores(text)


def classify_sentiment(compound: float) -> str:
    """
    Classify a compound score into a human-readable label.

    Thresholds follow VADER's published recommendations:
        compound >= 0.05  → Positive
        compound <= -0.05 → Negative
        otherwise         → Neutral
    """
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def score_dataframe(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add VADER sentiment columns to a DataFrame.

    Adds columns: vader_neg, vader_neu, vader_pos, vader_compound, sentiment_label.

    Parameters
    ----------
    df       : DataFrame containing the text column.
    text_col : Name of the column holding cleaned text.

    Returns
    -------
    DataFrame with sentiment columns added (original df is not modified).
    """
    df = df.copy()

    scores = df[text_col].apply(score_text)
    df["vader_neg"]      = scores.apply(lambda x: x["neg"])
    df["vader_neu"]      = scores.apply(lambda x: x["neu"])
    df["vader_pos"]      = scores.apply(lambda x: x["pos"])
    df["vader_compound"] = scores.apply(lambda x: x["compound"])
    df["sentiment_label"] = df["vader_compound"].apply(classify_sentiment)

    return df
