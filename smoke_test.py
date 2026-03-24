import os
import sys

import pandas as pd


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) or "."))

from shared.utils.sentiment_helpers import classify_sentiment, score_dataframe, score_text
from shared.utils.text_cleaning import batch_clean, clean_text


print("shared imports OK")

df = pd.read_csv("data/processed/scored_data.csv", parse_dates=["timestamp"])
print(f"CSV loaded: {df.shape[0]} rows x {df.shape[1]} cols")

samples = [
    "IEBC communication is too slow and the results delay is damaging trust.",
    "Peace volunteers in Kisumu are keeping the station calm and orderly.",
    "Observers say tallying continues while agents verify Form 34A uploads.",
]

for sample in samples:
    cleaned = clean_text(sample)
    scores = score_text(cleaned)
    label = classify_sentiment(scores["compound"])
    print(f"  [{label:8}] ({scores['compound']:+.3f}) {sample[:70]}")

required_columns = [
    "record_id",
    "timestamp",
    "topic",
    "region",
    "event_phase",
    "source_style",
    "sentiment_label",
    "vader_compound",
    "clean_text",
    "week_start",
]

missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    print(f"MISSING COLUMNS: {missing_columns}")
else:
    print("all required columns present OK")

mini_df = pd.DataFrame({"clean_text": batch_clean(samples)})
scored_df = score_dataframe(mini_df)
print(f"mini scoring OK: {len(scored_df)} rows")

print("SMOKE TEST PASSED")
