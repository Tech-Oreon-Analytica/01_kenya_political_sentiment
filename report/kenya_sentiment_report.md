<div style="font-family: Georgia, serif; color: #1a1a1a;">

<div style="text-align:center; padding: 40px 0 20px 0; border-bottom: 3px solid #1a5276;">
<h1 style="font-size:2em; color:#1a5276; margin-bottom:6px;">
Kenya Public Discourse Sentiment Analysis
</h1>
<h2 style="font-size:1.2em; font-weight:normal; color:#555;">
Prototype Technical Report — 2022 Political Period
</h2>
<p style="color:#777; font-size:0.95em;">
Tech Oreon Analytica &nbsp;|&nbsp; Prepared for: Government Stakeholder Engagement &nbsp;|&nbsp; 2025<br/>
<strong>PROTOTYPE — Synthetic Dataset Demo</strong>
</p>
</div>

<div style="page-break-before: always;"></div>

## 1. Executive Summary

This report presents a sentiment analysis prototype developed by Tech Oreon Analytica
to demonstrate the capability of computational text analytics on Kenyan public discourse.
The prototype covers the 2022 general election period (January–August 2022) — one of the
most analytically significant periods in recent Kenyan political history.

The system ingests text-based public discourse data, applies automated sentiment scoring,
and surfaces structured insights through an interactive web dashboard. Key capabilities
demonstrated include:

- **Real-time sentiment classification** into Positive, Neutral, and Negative categories
- **Topic-level sentiment breakdowns** across Elections, Economy, Security, Healthcare, and Governance
- **Time-series trend visualisation** showing how sentiment evolved month-by-month
- **Regional discourse mapping** across 10 major Kenyan urban centres
- **Live text analysis** — any text typed into the tool is scored instantly

**Prototype scope:** This demonstration uses a 500-record synthetic dataset, clearly
labelled throughout. It is designed to illustrate methodology and system capability.
Production deployment would integrate real data pipelines from social media, news, or
survey platforms.

---

## 2. Objective

The primary objective of this prototype is to answer a practical question facing
government communications and policy offices:

> *Can we systematically and cost-effectively monitor public sentiment on key issues,
> and translate that signal into actionable intelligence?*

This prototype answers **yes** — and demonstrates one credible path to doing so.

Specific analytical objectives:

1. Classify the sentiment of public discourse records as Positive, Neutral, or Negative
2. Identify which topics attract the most negative or positive discourse
3. Track how sentiment shifts over time — particularly around key political events
4. Provide a reusable, deployable tool that non-technical stakeholders can operate

---

## 3. Data Description

### 3.1 Dataset Overview

| Attribute | Detail |
|---|---|
| **Source** | Synthetic demonstration dataset |
| **Size** | 500 records |
| **Period** | 1 January 2022 – 31 August 2022 |
| **Format** | CSV (text, timestamp, topic, region) |
| **Language** | English |
| **Generation method** | Rule-based template generation with controlled topic/sentiment distribution |

### 3.2 Topic Distribution

The dataset covers five thematic domains weighted to reflect the political salience of
the 2022 election period:

| Topic | Approx. Share | Rationale |
|---|---|---|
| Elections | 35% | Dominant topic during 2022 general election cycle |
| Economy | 25% | Cost-of-living pressures a persistent concern |
| Governance | 13% | Accountability and devolution discourse |
| Security | 15% | North-East and election security concerns |
| Healthcare | 12% | Ongoing NHIF and strikes discourse |

### 3.3 Geographic Coverage

Records span 10 major Kenyan urban and peri-urban centres: Nairobi, Mombasa, Kisumu,
Nakuru, Eldoret, Thika, Machakos, Nyeri, Garissa, and Kakamega.

### 3.4 Important Data Notice

> **This is a synthetic demonstration dataset.** No real tweets, social media posts,
> news articles, or real individuals are represented. The dataset was purpose-built
> to illustrate the structure, range, and analytical potential of a real deployment.
> All findings in this report are illustrative of what the system *can* produce —
> they do not constitute conclusions on actual public opinion.

In a production deployment, the same pipeline can ingest:
- Twitter/X API data (filtered by keyword, location, or account)
- Facebook public page comments
- News article text (scraped with appropriate permissions)
- Survey open-text responses
- Government public consultation submissions

---

## 4. Methodology

### 4.1 Text Preprocessing

Raw text undergoes a five-step cleaning pipeline before sentiment scoring:

1. **URL removal** — strips all hyperlinks (http/https/www)
2. **Mention removal** — strips `@username` handles
3. **Hashtag expansion** — converts `#Elections2022` → `Elections2022`
   (preserves topical signal, which VADER can score)
4. **Emoji normalisation** — removes Unicode emoji; retains ASCII emoticons (`:)`, `:(`)
   which VADER's lexicon handles natively
5. **Whitespace normalisation** — collapses multiple spaces; strips leading/trailing whitespace

### 4.2 Sentiment Scoring — VADER

The prototype uses **VADER (Valence Aware Dictionary and sEntiment Reasoner)**, a
widely validated, open-source sentiment analysis tool developed at Georgia Tech and
calibrated specifically for social media text.

VADER produces four scores for each record:

| Score | Range | Meaning |
|---|---|---|
| `pos` | 0–1 | Proportion of positive-valence content |
| `neg` | 0–1 | Proportion of negative-valence content |
| `neu` | 0–1 | Proportion of neutral content |
| `compound` | −1 to +1 | Normalised aggregate score (primary classification score) |

**Classification rule** (per VADER's published recommendations):

- **Positive**: compound score ≥ +0.05
- **Negative**: compound score ≤ −0.05
- **Neutral**: −0.05 < compound < +0.05

### 4.3 Why VADER?

VADER was selected over heavier machine-learning alternatives for three reasons:

1. **Speed** — no model training required; runs on any machine in seconds
2. **Transparency** — lexicon-based; scores are fully explainable to non-technical audiences
3. **Social media optimisation** — specifically calibrated for informal, short-text content
   including political and news commentary

For production scaling, VADER can be combined with transformer-based models
(e.g., Afro-BERT or mBERT fine-tuned on Kenyan data) for higher accuracy,
especially on Kiswahili and code-switched content.

---

## 5. Key Findings

*Note: The following findings are derived from the synthetic prototype dataset.
They are presented to illustrate the output format and analytical depth of the system —
not to draw conclusions on real public opinion.*

### 5.1 Overall Sentiment Distribution

Across 500 synthetic records:

| Sentiment | Count | Share |
|---|---|---|
| Positive | ~175 | ~35% |
| Neutral | ~150 | ~30% |
| Negative | ~175 | ~35% |

The near-balanced split reflects the contested, polarised nature of discourse during
an election period — consistent with what real social media analysis of political events
typically produces.

### 5.2 Sentiment by Topic

**Elections** generated the most polarised sentiment, with a roughly equal
positive/negative split reflecting the credibility debate around IEBC and the
competing narratives of democratic success versus electoral fraud.

**Economy** showed a dominant negative sentiment pattern, consistent with the
real-world public frustration around fuel prices, inflation, and unemployment
that characterised this period.

**Governance** showed predominantly negative sentiment, reflecting deep-seated
concerns about corruption and accountability.

**Healthcare** was also largely negative, aligned with real-world reporting of
doctors' strikes and drug shortages in public hospitals.

**Security** showed a more mixed picture — positive coverage of professional
election security deployment alongside negative discourse on Northern Kenya insecurity.

### 5.3 Time Trend

The monthly trend reveals a pattern consistent with election-period dynamics:

- **January–April 2022**: Relatively balanced, pre-election discourse
- **May–July 2022**: Increased negative sentiment as campaign tensions escalate
- **August 2022**: Sharp spike in both positive and negative discourse on election day
  (9 August 2022) — reflecting the simultaneous hope and controversy that characterised
  the actual election

This temporal pattern validates the system's ability to surface event-driven sentiment
shifts — one of the most valuable capabilities for a real-time monitoring deployment.

### 5.4 Regional Variation

Nairobi and Mombasa dominated discourse volume (urban digital access bias).
Kisumu and Nakuru showed higher concentration of election-related records,
consistent with their status as politically significant regions.

---

## 6. Strategic Relevance

A production version of this system offers three high-value applications to government:

### 6.1 Public Communication Monitoring
Real-time tracking of how specific policies, announcements, or decisions land with
the public. A sentiment spike on a policy announcement can be detected within hours —
enabling communications teams to prepare responsive messaging.

### 6.2 Policy Feedback Tracking
Analysis of open-text submissions, public consultations, and social media commentary
on specific Bills or policy proposals — automatically classified and summarised.
This replaces weeks of manual qualitative coding with hours of automated analysis.

### 6.3 Early Warning on Public Tension
Sustained negative sentiment trends on specific topics (e.g., cost of living, security)
can serve as an early warning indicator, enabling proactive policy communication
before issues escalate.

---

## 7. Limitations

The following limitations apply to this prototype and to any social media sentiment
system. They are disclosed transparently to ensure informed interpretation.

| Limitation | Implication |
|---|---|
| **Sentiment ≠ public opinion** | Social media reflects active, vocal users — not a representative population sample. High-volume negative sentiment may reflect a motivated minority. |
| **English-language bias** | VADER is optimised for English. Kiswahili, Sheng, and code-switching (common in Kenyan public discourse) reduce accuracy. A production system requires multilingual models. |
| **Sarcasm and irony** | Rule-based models frequently misclassify ironic statements. "Oh yes, the government has done a *wonderful* job" would likely score as positive. |
| **Short-text limitations** | Very short texts (< 5 words) produce unreliable scores. |
| **Social media demographic skew** | Twitter/X users skew urban, younger, and educated — not representative of rural Kenya or older demographics. |
| **Synthetic data** | This prototype's findings reflect generated data, not real discourse. Production results will differ. |

---

## 8. Next Steps and Scaling Potential

### Phase 1 — Pilot Deployment (1–3 months)
- Connect to Twitter/X API (Filtered Stream endpoint) with Kenyan political keywords
- Integrate 2–3 news sources via RSS/web scraping
- Deploy on Streamlit Cloud or a government-approved cloud environment

### Phase 2 — Language Expansion (3–6 months)
- Fine-tune or integrate a multilingual model (mBERT, AfroXLM-R) on Kiswahili
- Develop a Sheng/code-switch vocabulary extension for VADER
- Validate accuracy against human-labelled Kenyan political tweets

### Phase 3 — Production System (6–12 months)
- Build a scheduled data ingestion pipeline (daily or hourly)
- Add user authentication and role-based access control
- Integrate with DHIS2 or government data infrastructure where applicable
- Develop a report-generation module (automated PDF briefings)

**Estimated cost to scale to Phase 1:** Minimal infrastructure cost if hosted on cloud.
The primary investment is data access (Twitter API tier) and a small team of 1–2
data engineers.

---

## 9. About This Prototype

**Developed by:** Tech Oreon Analytica  
**Technical lead:** Nichodemus Werre Amollo (Georgetown University gui2de)  
**Stack:** Python · VADER · Streamlit · Plotly  
**Version:** 1.0 Prototype  
**Status:** Ready for stakeholder demonstration

*This report and the accompanying dashboard are intended exclusively for demonstration
and capability assessment purposes. All data is synthetic. No findings should be
cited or disseminated as evidence of real public opinion.*

</div>
