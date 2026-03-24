"""
sentiment_pipeline.py
---------------------
End-to-end pipeline: load raw data → clean → score → save processed CSV.

Run this script once before launching the Streamlit app:
    python src/sentiment_pipeline.py

Outputs:
    data/processed/scored_data.csv

Author : Nichodemus Werre Amollo
"""

import sys
import os
import pandas as pd

# ── Path setup: allow imports from shared utilities ───────────────────────────
# Walks two directories up from src/ to the repo root, then into shared/
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _repo_root)

from shared.utils.text_cleaning    import batch_clean
from shared.utils.sentiment_helpers import score_dataframe


# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = os.path.join(os.path.dirname(__file__), "..", "data", "raw",  "kenya_political_sample.csv")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "scored_data.csv")


def run_pipeline() -> pd.DataFrame:
    """
    Execute the full preprocessing and sentiment scoring pipeline.

    Steps:
    1. Load raw CSV.
    2. Validate required columns are present.
    3. Clean text (URLs, mentions, hashtag expansion, lowercase).
    4. Score with VADER.
    5. Derive month column for time-series analysis.
    6. Save and return processed DataFrame.
    """

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run `python data/raw/generate_sample_data.py` first."
        )

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} records.")

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    required_cols = {"record_id", "text", "timestamp", "topic", "region"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    # ── Step 3: Clean text ────────────────────────────────────────────────────
    print("Cleaning text...")
    df["clean_text"] = batch_clean(
        df["text"],
        remove_mentions = True,
        remove_urls     = True,
        expand_hashtags = True,  # #Elections → Elections (preserves VADER signal)
        lowercase       = True,
    )

    # ── Step 4: VADER scoring ─────────────────────────────────────────────────
    print("Running VADER sentiment scoring...")
    df = score_dataframe(df, text_col="clean_text")

    # ── Step 5: Derive time features ──────────────────────────────────────────
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
    df["week"]       = df["timestamp"].dt.isocalendar().week.astype(int)

    # ── Step 6: Save ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Pipeline Summary ───────────────────────────────────────")
    print(f"  Total records   : {len(df):,}")
    print(f"  Date range      : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"\n  Sentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print(f"\n  Topic breakdown:")
    print(df["topic"].value_counts().to_string())
    print("────────────────────────────────────────────────────────────\n")

    return df


if __name__ == "__main__":
    run_pipeline()
