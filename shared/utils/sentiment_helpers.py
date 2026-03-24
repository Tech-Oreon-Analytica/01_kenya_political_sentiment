"""
Shared helpers for domain-tuned VADER sentiment scoring.
"""

from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05

CUSTOM_LEXICON = {
    "amani": 2.2,
    "tulivu": 1.4,
    "peaceful": 2.3,
    "peacefully": 2.2,
    "credible": 1.5,
    "transparent": 1.8,
    "orderly": 1.8,
    "hopeful": 1.6,
    "verified": 1.2,
    "calm": 1.6,
    "rigged": -3.5,
    "rigging": -3.6,
    "fraud": -3.4,
    "irregularity": -2.3,
    "irregularities": -2.3,
    "bribery": -2.9,
    "intimidation": -2.8,
    "violence": -3.1,
    "delay": -1.5,
    "delays": -1.7,
    "delayed": -1.7,
    "missing": -1.8,
    "stalled": -1.8,
    "chaos": -2.8,
    "unsafe": -2.2,
    "partisan": -1.9,
    "unprepared": -1.9,
    "frustrated": -1.8,
    "tension": -1.8,
    "threatening": -2.6,
    "mistrust": -2.0,
    "suspicion": -1.4,
    "iebc": 0.0,
    "bomas": 0.0,
    "kiems": 0.0,
    "azimio": 0.0,
    "kenya": 0.0,
    "ruto": 0.0,
    "raila": 0.0,
    "chebukati": 0.0,
    "smartmatic": 0.0,
    "form34a": 0.0,
}

PHRASE_ADJUSTMENTS = {
    "everything is moving well": 0.28,
    "let peace win": 0.32,
    "keep peace": 0.22,
    "orderly and hopeful": 0.24,
    "patient and calm": 0.24,
    "cannot find their names": -0.42,
    "opened late": -0.22,
    "opened late and some voters left": -0.36,
    "left before materials were ready": -0.28,
    "manual register not available": -0.34,
    "results are taking too long": -0.28,
    "without a clear explanation": -0.24,
    "road blockage": -0.34,
    "avoid the area": -0.26,
    "too few officials": -0.34,
    "acting partisan": -0.28,
    "mixed messaging": -0.20,
    "unanswered questions": -0.22,
    "agents are complaining": -0.18,
    "trust is getting damaged": -0.48,
    "not explaining what is happening": -0.26,
    "confidence is dropping fast": -0.52,
    "hard to calm voters": -0.40,
    "isolated delays under review": 0.55,
    "verification is still underway": 0.25,
    "no constituency should be skipped": 0.20,
}

KEYWORD_RULES = [
    (("kiems", "fail"), -0.30),
    (("kiems", "failing"), -0.30),
    (("kiems", "slow"), -0.16),
    (("queue", "hours"), -0.20),
    (("results", "delay"), -0.24),
    (("results", "delayed"), -0.24),
    (("peace", "message"), 0.18),
    (("peaceful", "queues"), 0.22),
    (("forms", "verify"), 0.12),
]


_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(CUSTOM_LEXICON)


def _clamp_compound(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _apply_domain_adjustments(text: str, compound: float) -> float:
    lowered = text.lower()
    adjustment = 0.0

    for phrase, delta in PHRASE_ADJUSTMENTS.items():
        if phrase in lowered:
            adjustment += delta

    for keywords, delta in KEYWORD_RULES:
        if all(keyword in lowered for keyword in keywords):
            adjustment += delta

    if any(marker in lowered for marker in ("headline:", "media update:", "observers say")) and abs(compound) < 0.30:
        adjustment *= 0.85

    return _clamp_compound(compound + adjustment)


def score_text(text: str) -> dict:
    """
    Score a single text string using VADER plus domain-specific adjustments.
    """
    if not isinstance(text, str) or not text.strip():
        return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

    scores = _analyzer.polarity_scores(text)
    scores["compound"] = _apply_domain_adjustments(text, scores["compound"])
    return scores


def classify_sentiment(compound: float) -> str:
    """
    Map a compound sentiment score to Positive, Neutral, or Negative.
    """
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    if compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def score_dataframe(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """
    Add VADER sentiment columns to a DataFrame.
    """
    df = df.copy()
    scores = df[text_col].apply(score_text)
    df["vader_neg"] = scores.apply(lambda item: item["neg"])
    df["vader_neu"] = scores.apply(lambda item: item["neu"])
    df["vader_pos"] = scores.apply(lambda item: item["pos"])
    df["vader_compound"] = scores.apply(lambda item: item["compound"])
    df["sentiment_label"] = df["vader_compound"].apply(classify_sentiment)
    return df
