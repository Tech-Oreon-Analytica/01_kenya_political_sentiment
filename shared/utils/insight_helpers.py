"""
Helpers for generating analyst-facing questions and hypotheses from the dataset.
"""

from __future__ import annotations

import pandas as pd


def _safe_mode(series: pd.Series, fallback: str = "N/A") -> str:
    if series.empty:
        return fallback
    modes = series.mode()
    return str(modes.iloc[0]) if not modes.empty else fallback


def build_analyst_brief(df: pd.DataFrame) -> dict:
    """
    Build a concise analyst brief from the currently filtered dataset.
    """
    if df.empty:
        return {
            "most_negative_topic": "N/A",
            "most_negative_region": "N/A",
            "peak_phase": "N/A",
            "peak_day": "N/A",
            "hypotheses": [],
            "questions": [],
        }

    topic_stats = (
        df.groupby("topic")
        .agg(records=("record_id", "count"), avg_compound=("vader_compound", "mean"))
        .sort_values(["avg_compound", "records"], ascending=[True, False])
    )

    region_stats = (
        df.groupby("region")
        .agg(records=("record_id", "count"), avg_compound=("vader_compound", "mean"))
        .query("records >= 15")
        .sort_values(["avg_compound", "records"], ascending=[True, False])
    )

    phase_stats = (
        df.groupby("event_phase")
        .agg(records=("record_id", "count"), avg_compound=("vader_compound", "mean"))
        .sort_values("records", ascending=False)
    )

    most_negative_topic = topic_stats.index[0]
    most_negative_region = region_stats.index[0] if not region_stats.empty else _safe_mode(df["region"])
    peak_phase = phase_stats.index[0] if not phase_stats.empty else _safe_mode(df["event_phase"])
    peak_day = df["event_date"].value_counts().idxmax()

    polarity_mix = (
        df.groupby("topic")["sentiment_label"]
        .value_counts(normalize=True)
        .rename("share")
        .reset_index()
        .pivot(index="topic", columns="sentiment_label", values="share")
        .fillna(0.0)
    )
    polarity_mix["polarization"] = 1.0 - polarity_mix.get("Neutral", 0.0)
    most_polarized_topic = polarity_mix["polarization"].sort_values(ascending=False).index[0]

    hypotheses = [
        f"{most_negative_topic} is the main driver of negative discourse in this slice and should be treated as the lead risk theme.",
        f"{most_negative_region} warrants closer review because it combines meaningful volume with the weakest sentiment balance.",
        f"{most_polarized_topic} appears to be the most polarized topic, suggesting competing reassurance and mistrust narratives are active at the same time.",
    ]

    questions = [
        f"What share of negative {most_negative_topic.lower()} records are operational complaints versus political trust complaints?",
        f"Is the pressure in {most_negative_region} concentrated around one event window or does it persist across phases?",
        f"What changed around {peak_day} during {peak_phase} that pushed conversation volume to its local high?",
    ]

    return {
        "most_negative_topic": most_negative_topic,
        "most_negative_region": most_negative_region,
        "peak_phase": peak_phase,
        "peak_day": peak_day,
        "hypotheses": hypotheses,
        "questions": questions,
    }
