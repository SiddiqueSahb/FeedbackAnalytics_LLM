"""
Dashboard - Customer Feedback Analytics
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import api_get, sidebar_status, SENTIMENT_COLORS

COLOR_MAP = SENTIMENT_COLORS

st.set_page_config(
    page_title="Dashboard | FeedbackIQ",
    page_icon="📊",
    layout="wide",
)

sidebar_status()

st.title("Customer Feedback Analytics Dashboard")
st.write(
    "Monitor customer sentiment, complaint trends and platform performance "
    "using AI-powered customer feedback analytics."
)

st.divider()


# Load analytics
with st.spinner("Loading dashboard..."):
    stats = api_get("/api/analytics/summary")
    by_platform = api_get("/api/analytics/sentiment-by-platform")
    rating_distribution = api_get("/api/analytics/rating-distribution")
    keywords = api_get("/api/analytics/keywords")
    trends = api_get("/api/analytics/trends")
    platforms = api_get("/api/analytics/platforms")


# Sidebar
st.sidebar.header("Filters")

if isinstance(platforms, list) and platforms:
    selected_platforms = st.sidebar.multiselect(
        "Platform",
        platforms,
        default=platforms,
    )
else:
    selected_platforms = []

selected_sentiment = st.sidebar.selectbox(
    "Sentiment",
    ["All", "positive", "negative", "neutral"],
)

st.sidebar.info(
    """
Filter the dashboard to compare platforms,
sentiment distribution and customer trends.
"""
)

# KPI cards
if stats:

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1, st.container(border=True):
        st.metric("Total Reviews", f"{stats.get('total_reviews',0):,}")

    with col2, st.container(border=True):
        st.metric("Average Rating", f"{stats.get('avg_rating',0):.1f} / 5")

    with col3, st.container(border=True):
        st.metric("Positive Reviews", f"{stats.get('positive_pct',0):.1f}%")

    with col4, st.container(border=True):
        st.metric("Negative Reviews", f"{stats.get('negative_pct',0):.1f}%")

    with col5, st.container(border=True):
        st.metric("Neutral Reviews", f"{stats.get('neutral_pct',0):.1f}%")

else:
    st.error(
        "Unable to load analytics. Please make sure the FastAPI backend is running."
    )

st.divider()

# Business Snapshot
with st.container(border=True):

    st.subheader("Business Snapshot")

    st.write(
        f"""
The dashboard currently analyses **{stats.get('total_reviews',0):,} customer reviews**
collected from multiple platforms.

Overall sentiment remains **{stats.get('positive_pct',0):.1f}% positive**, while
**{stats.get('negative_pct',0):.1f}%** of reviews indicate opportunities to improve
customer experience.
"""
    )

st.divider()

# Executive summary
positive = stats.get("positive_pct", 0)
negative = stats.get("negative_pct", 0)
rating = stats.get("avg_rating", 0)

platform_count = len(platforms)

summary = (
    f"A total of **{stats.get('total_reviews',0):,} customer reviews** "
    f"were analysed across **{platform_count} platforms**. "
    f"Overall sentiment is **{positive:.1f}% positive**, "
    f"with an average rating of **{rating:.1f}/5**. "
    f"Negative reviews account for **{negative:.1f}%** of customer feedback."
)
with st.container(border=True):

    st.subheader("Executive Summary")

    st.write(summary)

st.divider()

# Sentiment overview
if by_platform:

    df_platform = pd.DataFrame(by_platform)

    if selected_platforms:
        df_platform = df_platform[
            df_platform["platform"].isin(selected_platforms)
        ]

    if selected_sentiment != "All":
        df_platform = df_platform[
            df_platform["sentiment_label"] == selected_sentiment
        ]

    color_map = COLOR_MAP

    col1, col2 = st.columns([1.3, 1])

    with col1:

        st.subheader("Sentiment by Platform")

        fig = px.bar(
            df_platform,
            x="platform",
            y="count",
            color="sentiment_label",
            barmode="group",
            color_discrete_map=color_map,
            title="Sentiment by Platform",
            labels={
                "platform": "Platform",
                "count": "Reviews",
                "sentiment_label": "Sentiment",
            },
        )

        fig.update_layout(
            height=420,
            plot_bgcolor="white",
            legend_title="Sentiment",
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("Overall Sentiment")

        summary = (
            df_platform.groupby("sentiment_label")["count"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            summary,
            values="count",
            names="sentiment_label",
            hole=0.55,
            color="sentiment_label",
            color_discrete_map=color_map,
            title="Overall Sentiment",
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )

        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

st.divider()

# Rating distribution and keywords
if rating_distribution:

    st.subheader("Customer Ratings & Trending Keywords")

    col1, col2 = st.columns([1, 1])

    df_rating = pd.DataFrame(rating_distribution)

    with col1:

        fig = px.bar(
            df_rating,
            x="rating",
            y="count",
            text="count",
            color="rating",
            color_continuous_scale="RdYlGn",
            labels={
                "rating": "Star Rating",
                "count": "Reviews",
            },
        )

        fig.update_traces(textposition="outside")

        fig.update_layout(
            title="Rating Distribution",
            height=420,
            plot_bgcolor="white",
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        if keywords:

            df_keywords = (
                pd.DataFrame(keywords)
                .sort_values("count", ascending=True)
                .tail(15)
            )

            fig = px.bar(
                df_keywords,
                x="count",
                y="word",
                orientation="h",
                text="count",
                color="count",
                color_continuous_scale="Blues",
                labels={
                    "count": "Frequency",
                    "word": "Keyword",
                },
            )

            fig.update_layout(
                title="Most Frequent Keywords",
                height=420,
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=40, b=10),
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

st.divider()

# Monthly sentiment trends
if trends:

    st.subheader("Customer Sentiment Trends")

    df_trend = pd.DataFrame(trends)

    if not df_trend.empty:

        if selected_sentiment != "All":
            df_trend = df_trend[
                df_trend["sentiment_label"] == selected_sentiment
            ]

        color_map = COLOR_MAP

        fig = px.line(
            df_trend,
            x="month",
            y="count",
            color="sentiment_label",
            markers=True,
            color_discrete_map=color_map,
            title="Customer Sentiment Trends",
            labels={
                "month": "Month",
                "count": "Reviews",
                "sentiment_label": "Sentiment",
            },
        )

        fig.update_layout(
            height=420,
            plot_bgcolor="white",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=40, b=10),
            legend_title="Sentiment",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Trend data is not available for this dataset.")

st.divider()


# Business insights
st.subheader("Key Business Insights")

insight1, insight2 = st.columns(2)

with insight1, st.container(border=True):

    st.markdown("**Customer Satisfaction**")
    st.markdown(f"""
- Positive sentiment: **{stats.get('positive_pct',0):.1f}%**
- Average rating: **{stats.get('avg_rating',0):.1f}/5**
- Customers generally report a positive experience across platforms.
""")

with insight2, st.container(border=True):

    st.markdown("**Improvement Opportunities**")
    st.markdown(f"""
- Negative sentiment: **{stats.get('negative_pct',0):.1f}%**
- Review recurring complaints using Semantic Search.
- Use AI Review Analysis to identify actionable improvements.
""")

st.divider()


# Dashboard summary
st.subheader("Dashboard Summary")

st.write(
"""
FeedbackIQ combines machine learning, semantic search and large language models
to transform customer reviews into actionable business intelligence.

The dashboard enables users to:

- Monitor customer sentiment
- Compare platforms
- Identify complaint trends
- Explore reviews using semantic search
- Support data-driven business decisions
"""
)

st.divider()

st.caption(
"""
FeedbackIQ • AI-Powered Customer Feedback Analytics

Built using Python, FastAPI, Streamlit,
DistilBERT, BERTopic, BART,
Sentence Transformers, FAISS,
LangChain and Groq LLM.
"""
)


