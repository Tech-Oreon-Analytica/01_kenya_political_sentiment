"""
sentiment_pipeline.py
---------------------
End-to-end pipeline: load raw data -> clean -> score -> save processed CSV.
"""

from __future__ import annotations

import os
import sys

import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from shared.utils.sentiment_helpers import score_dataframe
from shared.utils.text_cleaning import batch_clean


RAW_DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "kenya_election_discourse_corpus.csv")
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "scored_data.csv")


def run_pipeline() -> pd.DataFrame:
    """
    Execute the preprocessing and sentiment scoring pipeline.

    Steps:
    1. Load the raw CSV.
    2. Validate the minimum required schema.
    3. Clean text.
    4. Score sentiment with domain-tuned VADER.
    5. Add time features for dashboard analysis.
    6. Save and return the processed frame.
    """

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_PATH}. "
            "Run `python data/raw/build_election_corpus.py` first."
        )

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} records.")

    required_cols = {"record_id", "text", "timestamp", "topic", "region"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_cols)}")

    print("Cleaning text...")
    df["clean_text"] = batch_clean(
        df["text"],
        remove_mentions=True,
        remove_urls=True,
        expand_hashtags=True,
        lowercase=True,
    )

    print("Running VADER sentiment scoring...")
    df = score_dataframe(df, text_col="clean_text")

    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["event_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["week_start"] = df["timestamp"].dt.to_period("W-SUN").apply(lambda period: period.start_time.strftime("%Y-%m-%d"))
    df["day_name"] = df["timestamp"].dt.strftime("%a")

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved processed data to: {PROCESSED_DATA_PATH}")

    print("\nPipeline Summary")
    print("----------------")
    print(f"Total records : {len(df):,}")
    print(f"Date range    : {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")
    print("\nSentiment distribution:")
    print(df["sentiment_label"].value_counts().to_string())
    print("\nTopic breakdown:")
    print(df["topic"].value_counts().to_string())
    print("----------------")

    return df


if __name__ == "__main__":
    run_pipeline()
