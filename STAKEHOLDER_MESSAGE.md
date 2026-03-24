# Stakeholder Handover Message
## Tech Oreon Analytica - Kenya Election Sentiment Analysis Prototype

**Subject:** Kenya 2022 Election Discourse Sentiment Analysis - Interactive Prototype

Dear [Recipient Name / Office],

Please find below the link to our Kenya Election Discourse Sentiment Analysis prototype,
developed by Tech Oreon Analytica as a working demonstration of how computational text
analysis can support election-period monitoring and analyst brief production.

**Live demo:** [INSERT STREAMLIT CLOUD LINK HERE]

## What the prototype shows

The dashboard allows a reviewer to:

- filter discourse by topic, sentiment, region, event phase, source style, and date
- inspect how sentiment changes over the election timeline
- compare issue areas such as counting and results, voting issues, and security concerns
- review an analyst brief with working hypotheses and investigation questions
- test new text in a live sentiment analyzer

## Important context

This build ships with a local project corpus aligned to documented 2022 Kenya election
discourse patterns and a production-ready CSV workflow. It is designed to show the
technical and analytical workflow clearly.

The strongest real-world replacement candidate identified during project review is the
Uchaguzi-2022 citizen-report dataset, which is access-controlled and can be slotted into
the same workflow once approval is secured.

## Supporting report

The written report is available at:

`report/kenya_sentiment_report.md`

It summarizes:

- project objective
- corpus design
- scoring methodology
- research footing
- example hypotheses
- next-step upgrade path

## Suggested next step

If you would like this prototype moved toward operational use, the next phase would be:

1. secure approved real-world data access
2. validate the model on a labeled Kenya election evaluation set
3. deploy scheduled ingestion and briefing output on top of the current app

Warm regards,

**Tech Oreon Analytica**  
[Contact Name | Title]  
[Email | Phone]
