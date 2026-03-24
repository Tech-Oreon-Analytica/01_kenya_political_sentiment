"""
Streamlit application for Kenya 2022 election discourse sentiment analysis.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from shared.utils.insight_helpers import build_analyst_brief
from shared.utils.sentiment_helpers import classify_sentiment, score_text
from shared.utils.text_cleaning import clean_text


st.set_page_config(
    page_title="Kenya Election Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "Positive": "#2f855a",
    "Neutral": "#718096",
    "Negative": "#c53030",
}

ACCENT = "#1f4e79"
CARD_BG = "#f7fafc"

PROCESSED_PATH = os.path.join(REPO_ROOT, "data", "processed", "scored_data.csv")
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "kenya_election_discourse_corpus.csv")
BUILDER_PATH = os.path.join(REPO_ROOT, "data", "raw", "build_election_corpus.py")
PIPELINE_PATH = os.path.join(REPO_ROOT, "src", "sentiment_pipeline.py")

PHASE_ORDER = [
    "Campaign Build-up",
    "Election Day",
    "Tallying & Results",
    "Post-Result Reaction",
]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load processed data, building the local corpus and scoring it if needed.
    """
    if not os.path.exists(PROCESSED_PATH):
        if not os.path.exists(RAW_PATH):
            subprocess.run([sys.executable, BUILDER_PATH], cwd=REPO_ROOT, check=True)
        subprocess.run([sys.executable, PIPELINE_PATH], cwd=REPO_ROOT, check=True)

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


def render_header(df: pd.DataFrame) -> None:
    date_min = df["timestamp"].min().date()
    date_max = df["timestamp"].max().date()
    st.title("Kenya 2022 Election Discourse Sentiment Analysis")
    st.caption("Tech Oreon Analytica | Streamlit + VADER | analyst-facing prototype")
    st.markdown(
        (
            f"<div style='background:{CARD_BG}; border-left:4px solid {ACCENT}; "
            "padding:12px 16px; border-radius:6px; margin-bottom:16px;'>"
            "<strong>Corpus note.</strong> "
            "This dashboard ships with a curated project corpus aligned to documented "
            "2022 Kenya election discourse patterns and a production-ready CSV schema. "
            "The current bundle is useful for demonstrating workflow, feature engineering, "
            f"and stakeholder reporting across the {date_min} to {date_max} election window. "
            "A larger real-world replacement candidate exists in the access-controlled "
            "Uchaguzi-2022 citizen-report dataset."
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    all_topics = sorted(df["topic"].unique().tolist())
    selected_topics = st.sidebar.multiselect("Topic", all_topics, default=all_topics)

    sentiments = ["Positive", "Neutral", "Negative"]
    selected_sentiments = st.sidebar.multiselect("Sentiment", sentiments, default=sentiments)

    all_phases = [phase for phase in PHASE_ORDER if phase in df["event_phase"].unique().tolist()]
    selected_phases = st.sidebar.multiselect("Event phase", all_phases, default=all_phases)

    all_sources = sorted(df["source_style"].unique().tolist())
    selected_sources = st.sidebar.multiselect("Source style", all_sources, default=all_sources)

    all_regions = sorted(df["region"].unique().tolist())
    selected_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    filtered = df[
        df["topic"].isin(selected_topics)
        & df["sentiment_label"].isin(selected_sentiments)
        & df["event_phase"].isin(selected_phases)
        & df["source_style"].isin(selected_sources)
        & df["region"].isin(selected_regions)
    ]

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["timestamp"] >= start_date) & (filtered["timestamp"] <= end_date)]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(filtered):,} records shown of {len(df):,} total")
    st.sidebar.caption("Research-shaped project corpus for the 2022 election period")
    return filtered


def render_kpis(df: pd.DataFrame) -> None:
    st.markdown("### Overview")
    if df.empty:
        st.warning("No records match the current filters.")
        return

    total_records = len(df)
    positive_share = round(100 * (df["sentiment_label"] == "Positive").mean(), 1)
    neutral_share = round(100 * (df["sentiment_label"] == "Neutral").mean(), 1)
    negative_share = round(100 * (df["sentiment_label"] == "Negative").mean(), 1)
    average_compound = df["vader_compound"].mean()
    peak_phase = df["event_phase"].value_counts().idxmax()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{total_records:,}")
    c2.metric("Positive Share", f"{positive_share}%")
    c3.metric("Neutral Share", f"{neutral_share}%")
    c4.metric("Negative Share", f"{negative_share}%")
    c5.metric("Most Active Phase", peak_phase)

    st.caption(f"Average compound score: {average_compound:+.3f}")
    st.markdown("---")


def render_analyst_brief(df: pd.DataFrame) -> None:
    brief = build_analyst_brief(df)
    st.markdown("### Analyst Brief")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Most Negative Topic", brief["most_negative_topic"])
    c2.metric("Most Negative Region", brief["most_negative_region"])
    c3.metric("Peak Phase", brief["peak_phase"])
    c4.metric("Peak Day", brief["peak_day"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Working Hypotheses")
        for item in brief["hypotheses"]:
            st.markdown(f"- {item}")
    with right:
        st.markdown("#### Questions To Investigate")
        for item in brief["questions"]:
            st.markdown(f"- {item}")

    st.markdown("---")


def chart_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment_label"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    chart_df = counts.reset_index()
    chart_df.columns = ["Sentiment", "Count"]
    chart_df["Share"] = (chart_df["Count"] / max(chart_df["Count"].sum(), 1) * 100).round(1)

    fig = px.bar(
        chart_df,
        x="Count",
        y="Sentiment",
        orientation="h",
        color="Sentiment",
        color_discrete_map=COLORS,
        text=chart_df["Share"].map(lambda value: f"{value}%"),
        title="Sentiment Distribution",
        labels={"Count": "Records"},
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=45, b=10))
    fig.update_traces(textposition="outside")
    return fig


def chart_avg_compound_by_topic(df: pd.DataFrame) -> go.Figure:
    topic_scores = (
        df.groupby("topic")["vader_compound"]
        .mean()
        .reset_index()
        .sort_values("vader_compound")
    )
    topic_scores["color"] = topic_scores["vader_compound"].apply(
        lambda value: COLORS["Positive"]
        if value >= 0.05
        else COLORS["Negative"]
        if value <= -0.05
        else COLORS["Neutral"]
    )

    fig = go.Figure(
        go.Bar(
            x=topic_scores["vader_compound"],
            y=topic_scores["topic"],
            orientation="h",
            marker_color=topic_scores["color"],
            text=topic_scores["vader_compound"].map(lambda value: f"{value:+.3f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Average Sentiment By Topic",
        height=300,
        margin=dict(l=10, r=60, t=45, b=10),
        xaxis_title="Average VADER compound score",
        showlegend=False,
    )
    return fig


def chart_time_trend(df: pd.DataFrame) -> go.Figure:
    weekly = (
        df.groupby(["week_start", "sentiment_label"])
        .size()
        .reset_index(name="Count")
        .sort_values("week_start")
    )
    fig = px.line(
        weekly,
        x="week_start",
        y="Count",
        color="sentiment_label",
        color_discrete_map=COLORS,
        markers=True,
        title="Weekly Sentiment Trend",
        labels={"week_start": "Week start", "sentiment_label": "Sentiment"},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=30), legend_title="Sentiment")
    return fig


def chart_topic_sentiment(df: pd.DataFrame) -> go.Figure:
    grouped = (
        df.groupby(["topic", "sentiment_label"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        grouped,
        x="topic",
        y="Count",
        color="sentiment_label",
        color_discrete_map=COLORS,
        barmode="group",
        title="Sentiment By Topic",
        labels={"topic": "Topic", "sentiment_label": "Sentiment"},
    )
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=50), legend_title="Sentiment")
    fig.update_xaxes(tickangle=-25)
    return fig


def chart_region_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.pivot_table(index="region", columns="topic", values="record_id", aggfunc="count", fill_value=0)
        .sort_index()
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            text=pivot.values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Record Volume: Region x Topic",
        height=360,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis_title="Topic",
        yaxis_title="Region",
    )
    return fig


def render_live_analyzer() -> None:
    st.markdown("### Live Text Analyzer")
    st.caption("Paste a short election-related message below to inspect the cleaning and sentiment output.")

    user_text = st.text_area(
        "Text to analyze",
        height=110,
        placeholder="Example: Voters in Kisumu say the KIEMS kit is failing and the queue is barely moving.",
    )

    if user_text.strip():
        cleaned = clean_text(user_text)
        scores = score_text(cleaned)
        label = classify_sentiment(scores["compound"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Label", label)
        c2.metric("Compound", f"{scores['compound']:+.3f}")
        c3.metric("Positive", f"{scores['pos']:.2f}")
        c4.metric("Negative", f"{scores['neg']:.2f}")

        st.code(cleaned, language="text")

    st.markdown("---")


def render_data_table(df: pd.DataFrame) -> None:
    st.markdown("### Sample Records")
    if df.empty:
        st.info("No records available for the current filter combination.")
        return

    max_rows = min(200, len(df))
    default_rows = min(25, max_rows)
    row_count = st.slider("Rows to display", min_value=10 if max_rows >= 10 else 1, max_value=max_rows, value=default_rows, step=5 if max_rows >= 10 else 1)

    display_columns = [
        "record_id",
        "timestamp",
        "event_phase",
        "topic",
        "region",
        "source_style",
        "sentiment_label",
        "vader_compound",
        "text",
    ]
    show_df = df[display_columns].head(row_count).copy()
    show_df["timestamp"] = show_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    show_df["vader_compound"] = show_df["vader_compound"].round(3)
    st.dataframe(show_df, use_container_width=True, height=400)
    st.caption(f"Showing {row_count} of {len(df):,} records.")
    st.markdown("---")


def render_methodology() -> None:
    with st.expander("Methodology, Corpus, and Limits", expanded=False):
        st.markdown(
            """
#### Modeling Approach

- Sentiment engine: VADER with Kenya-election domain vocabulary and phrase adjustments
- Text cleaning: URL removal, mention removal, hashtag expansion, lowercasing, whitespace normalization
- Scoring rule: Positive if compound >= 0.05, Negative if compound <= -0.05, otherwise Neutral

#### Corpus Design

The local corpus in this repo is aligned to the main operational phases of the 2022 Kenya election:

- Campaign build-up
- Election day
- Tallying and results
- Post-result reaction

The topic taxonomy and date window were shaped by publicly documented 2022 election discourse patterns and the
Uchaguzi-2022 citizen-report work, which describes a larger corpus of 14,169 categorized and geotagged reports
submitted between June 27 and August 29, 2022.

Useful public references:

- [Uchaguzi-2022 dataset access page](https://github.ushahidi.org/uchaguzi-ai/)
- [Uchaguzi-2022 paper on arXiv](https://arxiv.org/abs/2412.13098)
- [Meltwater 2022 Kenya election social/media infographic](https://www.meltwater.com/en/resources/kenya-election-infographic)

#### Important Limits

- This dashboard is a workflow demonstration, not a representative measure of all Kenyan public opinion.
- English-first sentiment models still struggle with sarcasm, deeper context, and code-switched English-Swahili text.
- The bundled project corpus should be replaced with approved live or licensed data for operational use.
- Negative sentiment should be interpreted as issue intensity, not automatically as electoral preference.
            """
        )


def render_footer() -> None:
    st.markdown(
        (
            "<hr/>"
            "<div style='text-align:center; color:#666; font-size:0.85rem; padding:8px 0;'>"
            "Tech Oreon Analytica | Kenya Election Sentiment Analysis | Portfolio prototype"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    with st.spinner("Loading data..."):
        df = load_data()

    if df.empty:
        st.error("No data available. Rebuild the local corpus and rerun the pipeline.")
        st.code("python data/raw/build_election_corpus.py\npython src/sentiment_pipeline.py")
        return

    render_header(df)
    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning("No records match the current filter combination.")
        return

    render_kpis(filtered_df)
    render_analyst_brief(filtered_df)

    st.markdown("### Visualizations")
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.plotly_chart(chart_sentiment_distribution(filtered_df), use_container_width=True)
    with row1_right:
        st.plotly_chart(chart_avg_compound_by_topic(filtered_df), use_container_width=True)

    st.plotly_chart(chart_time_trend(filtered_df), use_container_width=True)

    row2_left, row2_right = st.columns(2)
    with row2_left:
        st.plotly_chart(chart_topic_sentiment(filtered_df), use_container_width=True)
    with row2_right:
        st.plotly_chart(chart_region_heatmap(filtered_df), use_container_width=True)

    st.markdown("---")
    render_live_analyzer()
    render_data_table(filtered_df)
    render_methodology()
    render_footer()


if __name__ == "__main__":
    main()
