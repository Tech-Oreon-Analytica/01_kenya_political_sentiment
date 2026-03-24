"""
app.py
------
Streamlit web application: Kenya Political Discourse Sentiment Analysis Prototype
Tech Oreon Analytica — Demo for Government Stakeholders

Author : Nichodemus Werre Amollo | Georgetown University gui2de
Version: 1.0.0

Run:
    streamlit run app.py
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_repo_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _repo_root)

from shared.utils.text_cleaning     import clean_text
from shared.utils.sentiment_helpers import score_text, classify_sentiment

# ── App configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Kenya Sentiment Analysis | Tech Oreon Analytica",
    page_icon   = "🇰🇪",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Colour palette (professional, Kenya-inspired) ─────────────────────────────
COLORS = {
    "Positive" : "#2ecc71",   # green
    "Neutral"  : "#95a5a6",   # grey
    "Negative" : "#e74c3c",   # red
}

ACCENT      = "#1a5276"   # deep navy
BG_CARD     = "#f8f9fa"

# ── Data paths ────────────────────────────────────────────────────────────────
PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "data", "processed", "scored_data.csv")
RAW_PATH       = os.path.join(os.path.dirname(__file__), "data", "raw", "kenya_political_sample.csv")
GENERATOR_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "generate_sample_data.py")
PIPELINE_PATH  = os.path.join(os.path.dirname(__file__), "src", "sentiment_pipeline.py")


# ── Data loading with auto-generation fallback ────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Load processed sentiment data.
    If scored_data.csv does not exist, auto-generate sample data and run pipeline.
    Caches the result so it is not reloaded on every interaction.
    """
    if not os.path.exists(PROCESSED_PATH):
        # Auto-generate: run the generator then the pipeline so the app is self-contained
        import subprocess
        if not os.path.exists(RAW_PATH):
            subprocess.run([sys.executable, GENERATOR_PATH],
                           cwd=os.path.dirname(__file__), check=True)
        subprocess.run([sys.executable, PIPELINE_PATH],
                       cwd=os.path.dirname(__file__), check=True)

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])
    df["year_month"] = df["timestamp"].dt.to_period("M").astype(str)
    df["month_name"] = df["timestamp"].dt.strftime("%b %Y")
    return df


# ── Header ────────────────────────────────────────────────────────────────────
def render_header() -> None:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🇰🇪 Kenya Political Discourse — Sentiment Analysis")
        st.markdown(
            """
            <div style='background:#eaf2ff; border-left:4px solid #1a5276;
                        padding:10px 16px; border-radius:4px; margin-bottom:6px;'>
            <strong>⚠️ PROTOTYPE NOTICE</strong> — This is a demonstration prototype
            built on a synthetic sample dataset (500 records, Jan–Aug 2022 period).
            It is intended to illustrate analytical capability, not to report
            conclusions on real public opinion. Methodology is fully transparent below.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style='background:#1a5276; color:white; padding:14px 18px;
                        border-radius:6px; text-align:center; margin-top:14px;'>
            <strong>Tech Oreon Analytica</strong><br/>
            <small>Sentiment Intelligence Demo</small><br/>
            <small>v1.0 · 2025</small>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ── Sidebar filters ───────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return filtered DataFrame."""
    st.sidebar.header("🔍 Filters")

    # Topic filter
    all_topics = sorted(df["topic"].unique().tolist())
    sel_topics = st.sidebar.multiselect(
        "Topic",
        options  = all_topics,
        default  = all_topics,
        help     = "Filter by discourse topic"
    )

    # Sentiment filter
    all_sentiments = ["Positive", "Neutral", "Negative"]
    sel_sentiments = st.sidebar.multiselect(
        "Sentiment",
        options = all_sentiments,
        default = all_sentiments,
    )

    # Region filter
    all_regions = sorted(df["region"].unique().tolist())
    sel_regions = st.sidebar.multiselect(
        "Region",
        options = all_regions,
        default = all_regions,
    )

    # Date range slider
    min_date = df["timestamp"].min().date()
    max_date = df["timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value  = (min_date, max_date),
        min_value = min_date,
        max_value = max_date,
    )

    # Apply filters
    filtered = df[
        df["topic"].isin(sel_topics) &
        df["sentiment_label"].isin(sel_sentiments) &
        df["region"].isin(sel_regions)
    ]

    # Date filter (handle both single and range date_input)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        filtered = filtered[
            (filtered["timestamp"] >= start_d) & (filtered["timestamp"] <= end_d)
        ]

    st.sidebar.markdown("---")
    st.sidebar.caption(f"**{len(filtered):,}** records shown of **{len(df):,}** total")
    st.sidebar.caption("📦 Data: Synthetic demo dataset (2022 election period)")
    return filtered


# ── KPI cards ─────────────────────────────────────────────────────────────────
def render_kpis(df: pd.DataFrame) -> None:
    """Display top-level KPI metrics in a 5-column row."""
    st.markdown("### 📊 Overview")
    n = len(df)
    if n == 0:
        st.warning("No records match the current filters.")
        return

    pct_pos  = round(100 * (df["sentiment_label"] == "Positive").sum() / n, 1)
    pct_neu  = round(100 * (df["sentiment_label"] == "Neutral").sum()  / n, 1)
    pct_neg  = round(100 * (df["sentiment_label"] == "Negative").sum() / n, 1)
    avg_comp = round(df["vader_compound"].mean(), 3)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",      f"{n:,}")
    c2.metric("% Positive",         f"{pct_pos}%",  delta=None)
    c3.metric("% Neutral",          f"{pct_neu}%",  delta=None)
    c4.metric("% Negative",         f"{pct_neg}%",  delta=None)
    c5.metric("Avg. Compound Score", f"{avg_comp:+.3f}",
              help="VADER compound: −1 (most negative) to +1 (most positive)")
    st.markdown("---")


# ── Charts ────────────────────────────────────────────────────────────────────
def chart_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of sentiment label counts."""
    counts = df["sentiment_label"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    counts["Pct"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)
    # Ensure ordering: Positive, Neutral, Negative
    order = ["Positive", "Neutral", "Negative"]
    counts["Sentiment"] = pd.Categorical(counts["Sentiment"], categories=order, ordered=True)
    counts = counts.sort_values("Sentiment")

    fig = px.bar(
        counts,
        x            = "Count",
        y            = "Sentiment",
        orientation  = "h",
        color        = "Sentiment",
        color_discrete_map = COLORS,
        text         = counts["Pct"].map(lambda v: f"{v}%"),
        title        = "Sentiment Distribution",
        labels       = {"Count": "Number of Records"},
    )
    fig.update_layout(showlegend=False, height=280, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_traces(textposition="outside")
    return fig


def chart_time_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of monthly sentiment counts over time."""
    monthly = (
        df.groupby(["year_month", "sentiment_label"])
        .size()
        .reset_index(name="Count")
        .sort_values("year_month")
    )
    fig = px.line(
        monthly,
        x       = "year_month",
        y       = "Count",
        color   = "sentiment_label",
        color_discrete_map = COLORS,
        markers = True,
        title   = "Monthly Sentiment Trend (Jan–Aug 2022)",
        labels  = {"year_month": "Month", "Count": "Records", "sentiment_label": "Sentiment"},
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=40),
                      xaxis_tickangle=-30, legend_title="Sentiment")
    return fig


def chart_topic_sentiment(df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart: sentiment breakdown by topic."""
    topic_sent = (
        df.groupby(["topic", "sentiment_label"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        topic_sent,
        x          = "topic",
        y          = "Count",
        color      = "sentiment_label",
        color_discrete_map = COLORS,
        barmode    = "group",
        title      = "Sentiment by Topic",
        labels     = {"topic": "Topic", "Count": "Records", "sentiment_label": "Sentiment"},
    )
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=40),
                      legend_title="Sentiment")
    return fig


def chart_avg_compound_by_topic(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of average VADER compound score per topic."""
    avg_scores = (
        df.groupby("topic")["vader_compound"]
        .mean()
        .reset_index()
        .rename(columns={"vader_compound": "Avg Compound"})
        .sort_values("Avg Compound")
    )
    avg_scores["Colour"] = avg_scores["Avg Compound"].apply(
        lambda v: COLORS["Positive"] if v >= 0.05 else
                  COLORS["Negative"] if v <= -0.05 else COLORS["Neutral"]
    )
    fig = go.Figure(go.Bar(
        x           = avg_scores["Avg Compound"],
        y           = avg_scores["topic"],
        orientation = "h",
        marker_color = avg_scores["Colour"],
        text        = avg_scores["Avg Compound"].map(lambda v: f"{v:+.3f}"),
        textposition = "outside",
    ))
    fig.update_layout(
        title   = "Average Sentiment Score by Topic",
        xaxis   = dict(title="Avg. VADER Compound Score", zeroline=True,
                       zerolinecolor="black", zerolinewidth=1),
        height  = 280,
        margin  = dict(l=10, r=60, t=40, b=10),
        showlegend = False,
    )
    return fig


def chart_region_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: topic × region by record count."""
    pivot = df.pivot_table(index="region", columns="topic", values="record_id",
                           aggfunc="count", fill_value=0)
    fig = go.Figure(go.Heatmap(
        z           = pivot.values,
        x           = pivot.columns.tolist(),
        y           = pivot.index.tolist(),
        colorscale  = "Blues",
        text        = pivot.values,
        texttemplate = "%{text}",
    ))
    fig.update_layout(
        title  = "Record Volume: Region × Topic",
        height = 340,
        margin = dict(l=10, r=10, t=40, b=10),
        xaxis_title = "Topic",
        yaxis_title = "Region",
    )
    return fig


# ── Live text analyser ────────────────────────────────────────────────────────
def render_live_analyzer() -> None:
    """Interactive section: user types any text and gets instant VADER scoring."""
    st.markdown("### 🧪 Try It: Live Text Analyser")
    st.caption("Type any text below to see how VADER scores it in real time.")

    user_text = st.text_area(
        "Enter text to analyse",
        placeholder="e.g. The government has failed to address the rising cost of living.",
        height=100,
        key="live_text",
    )

    if user_text.strip():
        cleaned   = clean_text(user_text)
        scores    = score_text(cleaned)
        compound  = scores["compound"]
        label     = classify_sentiment(compound)
        label_color = {"Positive": "#2ecc71", "Neutral": "#95a5a6", "Negative": "#e74c3c"}[label]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Label",     label)
        c2.metric("Compound",  f"{compound:+.3f}")
        c3.metric("Positive",  f"{scores['pos']:.2f}")
        c4.metric("Negative",  f"{scores['neg']:.2f}")

        st.markdown(
            f"<div style='background:{label_color}22; border-left:4px solid {label_color};"
            f"padding:8px 14px; border-radius:4px; margin-top:6px;'>"
            f"<strong>Cleaned text:</strong> <em>{cleaned}</em></div>",
            unsafe_allow_html=True,
        )
    st.markdown("---")


# ── Data table ────────────────────────────────────────────────────────────────
def render_data_table(df: pd.DataFrame) -> None:
    """Display a paginated sample data table with key columns."""
    st.markdown("### 🗂️ Sample Records")

    display_cols = ["record_id", "timestamp", "topic", "region",
                    "sentiment_label", "vader_compound", "text"]
    available    = [c for c in display_cols if c in df.columns]

    n_show = st.slider("Rows to display", min_value=10, max_value=min(200, len(df)),
                       value=25, step=5)

    show_df = df[available].head(n_show).copy()
    show_df["vader_compound"] = show_df["vader_compound"].round(3)
    show_df["timestamp"]      = show_df["timestamp"].dt.strftime("%Y-%m-%d")

    st.dataframe(show_df, use_container_width=True, height=380)
    st.caption(f"Showing {n_show} of {len(df):,} records. Data: Synthetic demo dataset.")
    st.markdown("---")


# ── Methodology section ───────────────────────────────────────────────────────
def render_methodology() -> None:
    """Expandable methodology and limitations section."""
    with st.expander("📖 Methodology & Limitations", expanded=False):
        st.markdown("""
### Sentiment Analysis Methodology

**Tool:** VADER (Valence Aware Dictionary and sEntiment Reasoner)

VADER is a lexicon- and rule-based sentiment analysis tool specifically
calibrated for social media and short-text content. It returns four scores:

| Score      | Range | Meaning |
|------------|-------|---------|
| `pos`      | 0–1   | Proportion of positive sentiment |
| `neg`      | 0–1   | Proportion of negative sentiment |
| `neu`      | 0–1   | Proportion of neutral content |
| `compound` | −1 to +1 | Normalised aggregate sentiment |

**Classification thresholds** (per VADER's published recommendations):
- **Positive**: compound ≥ 0.05
- **Negative**: compound ≤ −0.05
- **Neutral**: −0.05 < compound < 0.05

---

### Text Preprocessing Steps

1. **URL removal** — strips `http://`, `https://`, and `www.*` links
2. **Mention removal** — strips `@user` handles
3. **Hashtag expansion** — converts `#Elections` → `Elections` (preserves topic signal for VADER)
4. **Emoji removal** — removes Unicode emoji (ASCII emoticons like `:)` preserved for VADER)
5. **Lowercasing** — normalises case
6. **Whitespace normalisation** — removes excess spaces

---

### Dataset

This prototype uses a **synthetic demonstration dataset** of 500 records generated
to reflect the structure and discourse themes of the 2022 Kenyan general election period
(January–August 2022). The dataset covers five topic domains: Elections, Economy,
Security, Healthcare, and Governance across 10 Kenyan regions.

> **This data does not represent real tweets, real individuals, or real events.**
> It exists solely to demonstrate the analytical pipeline.

---

### Limitations

| Limitation | Detail |
|-----------|--------|
| **Sentiment ≠ public opinion** | Social media discourse is not a representative sample of the population |
| **Language gap** | VADER is optimised for English; Kiswahili and Sheng require dedicated models |
| **Sarcasm/irony** | Rule-based models struggle with ironic or sarcastic language |
| **Context loss** | Short-text sentiment can miss broader context |
| **Social media bias** | Demographic skew toward urban, educated, connected users |
| **Synthetic data** | This prototype uses fabricated data; real-world results will differ |

---

### References

- Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text.* ICWSM-14.
        """)


# ── Footer ────────────────────────────────────────────────────────────────────
def render_footer() -> None:
    st.markdown(
        """
        <hr/>
        <div style='text-align:center; color:#888; font-size:0.82rem; padding:10px 0;'>
        <strong>Tech Oreon Analytica</strong> · Kenya Sentiment Analysis Prototype v1.0 ·
        Built with VADER + Streamlit ·
        <em>Prototype — Synthetic Data Only</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main application ──────────────────────────────────────────────────────────
def main() -> None:
    """
    Main Streamlit app entry point.
    Orchestrates all sections in order:
      header → data load → sidebar filters → KPIs →
      charts (2 rows) → live analyser → data table → methodology → footer
    """
    render_header()

    # Load data (cached after first run)
    with st.spinner("Loading data..."):
        df = load_data()

    if df.empty:
        st.error("No data found. Please run the data generator and pipeline first.")
        st.code("python data/raw/generate_sample_data.py\npython src/sentiment_pipeline.py")
        return

    # Apply sidebar filters → all downstream sections use filtered df
    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning("No records match the current filter combination. Adjust the sidebar filters.")
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    render_kpis(filtered_df)

    # ── Charts: row 1 ─────────────────────────────────────────────────────────
    st.markdown("### 📈 Visualisations")
    col_a, col_b = st.columns(2)

    with col_a:
        st.plotly_chart(chart_sentiment_distribution(filtered_df),
                        use_container_width=True)
    with col_b:
        st.plotly_chart(chart_avg_compound_by_topic(filtered_df),
                        use_container_width=True)

    # ── Charts: row 2 ─────────────────────────────────────────────────────────
    st.plotly_chart(chart_time_trend(filtered_df), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_topic_sentiment(filtered_df),
                        use_container_width=True)
    with col_d:
        st.plotly_chart(chart_region_heatmap(filtered_df),
                        use_container_width=True)

    st.markdown("---")

    # ── Live text analyser ────────────────────────────────────────────────────
    render_live_analyzer()

    # ── Data table ────────────────────────────────────────────────────────────
    render_data_table(filtered_df)

    # ── Methodology ───────────────────────────────────────────────────────────
    render_methodology()

    # ── Footer ────────────────────────────────────────────────────────────────
    render_footer()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
