# 01 - Kenya 2022 Election Discourse Sentiment Analysis
### Tech Oreon Analytica | Analyst-facing prototype for stakeholder review

## Overview

This project demonstrates a complete sentiment-analysis workflow for discourse around the
2022 Kenya election period. It packages:

- a research-shaped election corpus in CSV form
- a preprocessing and sentiment-scoring pipeline
- a Streamlit dashboard for filtering, analysis, and live scoring
- a written report and stakeholder handoff material

The focus is not just charting sentiment, but showing analytical thinking:

- which issues are driving negativity
- when pressure spikes during the election cycle
- where analysts should dig deeper next
- how the same project can later plug into approved real-world data

## Research footing

While inspecting available real-world data, the strongest directly relevant source I found was
the Uchaguzi-2022 citizen-report dataset from Ushahidi and collaborators:

- 14,169 categorized and geotagged election-related reports
- June 27, 2022 to August 29, 2022 coverage
- access-controlled rather than bundled for direct public redistribution

Useful references:

- [Uchaguzi-2022 dataset access page](https://github.ushahidi.org/uchaguzi-ai/)
- [Uchaguzi-2022 paper on arXiv](https://arxiv.org/abs/2412.13098)
- [Meltwater Kenya election infographic](https://www.meltwater.com/en/resources/kenya-election-infographic)

Because the real dataset requires an approval workflow, this repo ships with a local project corpus
that mirrors the working shape of a production pipeline and is aligned to documented 2022 election
topics, phases, and discourse patterns.

## Project structure

| File / Folder | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `src/sentiment_pipeline.py` | Clean and score the corpus |
| `data/raw/build_election_corpus.py` | Rebuild the local election corpus |
| `data/raw/kenya_election_discourse_corpus.csv` | Raw corpus |
| `data/processed/scored_data.csv` | Processed and sentiment-scored output |
| `shared/utils/text_cleaning.py` | Text preprocessing helpers |
| `shared/utils/sentiment_helpers.py` | Domain-tuned VADER scoring helpers |
| `shared/utils/insight_helpers.py` | Analyst-question and hypothesis helpers |
| `report/kenya_sentiment_report.md` | Portfolio/stakeholder report |
| `STAKEHOLDER_MESSAGE.md` | Handoff note for sharing the prototype |

## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the corpus and run the pipeline
```bash
python data/raw/build_election_corpus.py
python src/sentiment_pipeline.py
```

### 3. Launch the dashboard
```bash
streamlit run app.py
```

The app also supports an auto-build fallback: if the processed file is missing, it will rebuild
the raw corpus and rerun the pipeline automatically.

## What is different in this version

- The corpus now follows 2022 election phases rather than a generic political-period sample.
- Topic taxonomy is closer to real election operations and discourse:
  Public Opinion, Media Reports, Positive Events, Counting and Results, Security Issues,
  Voting Issues, Political Rallies, Polling Administration, and Staffing Issues.
- Sentiment scoring is tuned with Kenya-election vocabulary and phrase rules on top of VADER.
- The dashboard now generates working hypotheses and investigation questions from the filtered data.

## Analysis ideas this project supports

- Which topics drive the sharpest negative sentiment during tallying and results?
- Are certain regions consistently more negative, or only during one event window?
- Does peace messaging offset security-related tension in the same period?
- Are complaints primarily about trust, operations, or candidate mobilization?

## Limits

- The bundled corpus is a project-ready local dataset, not a substitute for approved live collection.
- Sentiment is a measure of tone in text, not a direct measure of vote intent or population opinion.
- English-first rule-based scoring still loses nuance on sarcasm and code-switched language.

## Next production step

The cleanest upgrade path is to keep the current schema and swap the local corpus for:

1. approved Uchaguzi-2022 access, or
2. approved live social/news ingestion with the same `record_id`, `text`, `timestamp`, `topic`, and `region` core fields.
