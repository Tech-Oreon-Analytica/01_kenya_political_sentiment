<div style="font-family: Georgia, serif; color: #1a1a1a;">

<div style="text-align:center; padding: 40px 0 20px 0; border-bottom: 3px solid #1f4e79;">
<h1 style="font-size:2em; color:#1f4e79; margin-bottom:6px;">
Kenya 2022 Election Discourse Sentiment Analysis
</h1>
<h2 style="font-size:1.2em; font-weight:normal; color:#555;">
Portfolio Prototype Report
</h2>
<p style="color:#777; font-size:0.95em;">
Tech Oreon Analytica | Prepared for stakeholder review | 2026
</p>
</div>

## 1. Executive Summary

This project demonstrates a full sentiment-analysis workflow for discourse around the
2022 Kenya election cycle. The prototype combines a local election corpus, a text-cleaning
and scoring pipeline, and a Streamlit dashboard that helps an analyst move from raw text
to questions, hypotheses, and stakeholder-ready summaries.

The strongest publicly documented real-world data source identified during project review
was the Uchaguzi-2022 citizen-report dataset, which covers 14,169 categorized and geotagged
election-related reports submitted between June 27, 2022 and August 29, 2022. Because that
dataset requires an access request rather than open redistribution, the repo currently ships
with a local project corpus shaped to the same operational problem space.

The goal of this build is therefore twofold:

1. show the technical workflow clearly and cleanly
2. show the reasoning style expected from a real sentiment and public-discourse analysis project

## 2. Project Objective

The prototype is built to answer a practical stakeholder question:

> Can a lightweight NLP workflow turn fast-moving election discourse into a structured
> analyst brief that highlights risk themes, pressure points, and follow-up questions?

This project argues that the answer is yes, provided the workflow is explicit about:

- corpus provenance
- topic design
- scoring limits
- what sentiment can and cannot claim

## 3. Corpus Design

### 3.1 Time Window

The local corpus is aligned to four election phases:

- Campaign Build-up
- Election Day
- Tallying and Results
- Post-Result Reaction

### 3.2 Topic Taxonomy

The corpus uses issue categories that are analytically useful for election monitoring:

- Public Opinion
- Media Reports
- Positive Events
- Counting and Results
- Security Issues
- Voting Issues
- Political Rallies
- Polling Administration
- Staffing Issues

### 3.3 Core Schema

The raw data keeps a production-friendly shape:

| Field | Purpose |
|---|---|
| `record_id` | unique record identifier |
| `text` | discourse content |
| `timestamp` | event time |
| `topic` | issue category |
| `region` | reporting region |

Additional metadata such as `event_phase`, `reference_event`, `source_style`,
and `language_style` support deeper filtering and hypothesis generation.

## 4. Methodology

### 4.1 Text Processing

The pipeline cleans raw text through:

- URL removal
- mention removal
- hashtag expansion
- lowercasing
- whitespace normalization

### 4.2 Sentiment Scoring

The project uses VADER as the base sentiment engine, then adds Kenya-election
domain tuning through:

- vocabulary adjustments for election language such as `rigged`, `violence`, `peaceful`, and `credible`
- phrase-level rules for operational complaints such as KIEMS failures, missing registers, and results delays

This keeps the model transparent and fast while improving domain fit over plain off-the-shelf VADER.

### 4.3 Analyst Layer

The dashboard does more than visualize scores. It also generates a compact analyst brief:

- most negative topic
- most negative region
- peak discussion phase
- peak discussion day
- working hypotheses
- investigation questions

This moves the output from descriptive charts toward a real analysis workflow.

## 5. Research Footing

The most relevant external references used to reshape this project were:

- Uchaguzi-2022 dataset access page: https://github.ushahidi.org/uchaguzi-ai/
- Uchaguzi-2022 paper: https://arxiv.org/abs/2412.13098
- Meltwater Kenya election social/media infographic:
  https://www.meltwater.com/en/resources/kenya-election-infographic

These references were used to align the repo more closely to:

- the actual election reporting window
- real issue clusters such as counting irregularities, voting issues, and peace messaging
- the observed importance of social and short-text discourse in the election cycle

## 6. Example Hypotheses This Workflow Supports

The dashboard is designed to help an analyst test ideas like:

1. Counting and results discourse is the main driver of negative sentiment during the tallying window.
2. Operational frustrations are more localized than broad opinion negativity and should be assessed region by region.
3. Positive-event and peace-message records help offset tension spikes but do not fully neutralize trust-related negativity.
4. The sharpest discourse spikes are tied to event windows, not steady background sentiment.

## 7. Important Interpretation Limits

| Limit | Why it matters |
|---|---|
| Sentiment is not vote intent | Tone in text does not equal support at the ballot |
| Social discourse is not the full population | Public conversation is biased toward connected and vocal users |
| English-first tooling loses nuance | Code-switching, sarcasm, and local idioms remain hard |
| A local project corpus is not operational evidence | Production use should plug into approved real-world data |

## 8. Recommended Next Step

The cleanest next move is not to redesign the dashboard. It is to keep the current
schema and replace the local corpus with one of the following:

1. approved access to Uchaguzi-2022
2. approved live ingestion from social, media, or public-feedback sources
3. a human-labeled Kenya election evaluation set for model validation

That would preserve the current app, pipeline, and analytic framing while upgrading the evidence base.

</div>
