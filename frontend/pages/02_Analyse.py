"""
Review Analysis — run a single review through the full FeedbackIQ pipeline:
sentiment, complaint category, similar past reviews and an LLM business
summary.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import api_get, api_post, sidebar_status, SENTIMENT_COLORS

st.set_page_config(
    page_title="Analyse | FeedbackIQ",
    page_icon="🧠",
    layout="wide",
)

sidebar_status()

st.title("AI Review Analysis")
st.write(
    "Paste a customer review below to see its predicted sentiment, complaint "
    "category, similar historical reviews and an AI-generated business summary."
)

st.divider()

# Model options come from the backend so the list can't drift out of sync
# with what the API actually supports.
model_options = api_get("/api/sentiment/models")

if not isinstance(model_options, list) or not model_options:
    model_options = [{"id": "distilbert", "name": "DistilBERT (fine-tuned)", "tier": ""}]

model_labels = {m["id"]: f"{m['name']} — {m['tier']}" for m in model_options}

col1, col2 = st.columns([3, 1])

with col1:
    review_text = st.text_area(
        "Customer review",
        height=160,
        placeholder="e.g. The battery stopped charging after one week and customer support never replied to my emails.",
    )

with col2:
    platform = st.selectbox("Platform (optional)", ["", "amazon", "yelp", "twitter_airline"])

    model_choices = list(model_labels.keys())
    default_index = (
        model_choices.index(st.session_state["last_model"])
        if st.session_state.get("last_model") in model_choices
        else 0
    )
    model_id = st.selectbox(
        "Sentiment model",
        options=model_choices,
        index=default_index,
        format_func=lambda x: model_labels[x],
    )
    st.session_state["last_model"] = model_id

analyse_clicked = st.button("Analyse Review", type="primary", use_container_width=True)

st.divider()

if not analyse_clicked:

    with st.container(border=True):
        st.markdown("**Results will appear here**")
        st.caption(
            "Paste a review above and click Analyse Review to see its sentiment, "
            "complaint category, similar historical reviews and an AI-generated "
            "business summary."
        )

if analyse_clicked:

    if len(review_text.strip()) < 3:
        st.warning("Please enter a review of at least 3 characters.")

    else:
        with st.spinner("Running the full analysis pipeline — this can take a few seconds..."):
            result = api_post(
                "/api/sentiment/analyse",
                {
                    "text": review_text,
                    "platform": platform or None,
                    "model": model_id,
                },
                timeout=90,
            )

        if result:

            sentiment = result.get("sentiment", {})
            analysis = result.get("analysis", {})

            # Sentiment
            st.subheader("Sentiment")

            sent_col1, sent_col2 = st.columns([1, 2])

            with sent_col1:
                st.metric(
                    "Predicted sentiment",
                    sentiment.get("label", "unknown").capitalize(),
                    f"{sentiment.get('confidence', 0):.0%} confidence",
                )

            with sent_col2:
                scores = sentiment.get("scores", {})
                if scores:
                    df_scores = pd.DataFrame(
                        {"Sentiment": list(scores.keys()), "Score": list(scores.values())}
                    )
                    fig = px.bar(
                        df_scores, x="Sentiment", y="Score",
                        color="Sentiment",
                        color_discrete_map=SENTIMENT_COLORS,
                        range_y=[0, 1],
                    )
                    fig.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Complaint category
            st.subheader("Complaint Category")

            st.info(f"**Primary category:** {result.get('category', 'N/A')}")

            categories = result.get("categories", [])
            if categories:
                df_cat = pd.DataFrame(categories)
                st.dataframe(df_cat, use_container_width=True, hide_index=True)

            st.divider()

            # LLM business analysis
            st.subheader("Business Analysis")

            if analysis.get("summary"):

                a1, a2 = st.columns(2)

                with a1:
                    with st.container(border=True):
                        st.markdown(f"**Summary**\n\n{analysis.get('summary', '')}")
                        st.markdown(f"**Business insight**\n\n{analysis.get('business_insight', '')}")
                        if analysis.get("keywords"):
                            st.markdown("**Keywords:** " + ", ".join(analysis["keywords"]))

                with a2:
                    with st.container(border=True):
                        st.markdown(f"**Severity:** {analysis.get('severity', 'N/A')}")
                        st.markdown(f"**Priority:** {analysis.get('priority', 'N/A')}")
                        st.markdown(f"**Department:** {analysis.get('department', 'N/A')}")

                with st.container(border=True):
                    st.markdown(f"**Executive summary**\n\n{analysis.get('executive_summary', '')}")

            else:
                st.warning(
                    "The LLM business analysis is unavailable — check that GROQ_API_KEY "
                    "is configured on the backend."
                )

            st.divider()

            # Similar reviews
            st.subheader("Similar Historical Reviews")

            similar = result.get("similar_reviews", [])

            if similar:
                df_similar = pd.DataFrame(similar)
                st.dataframe(df_similar, use_container_width=True, hide_index=True)
            else:
                st.caption("No similar reviews were found in the index.")

st.divider()

# Compare all models
with st.expander("⚖️ Compare all five sentiment models on this review"):

    if st.button("Run comparison"):

        if len(review_text.strip()) < 3:
            st.warning("Please enter a review above first.")

        else:
            with st.spinner("Scoring with all five models..."):
                comparison = api_post("/api/sentiment/compare", {"text": review_text})

            if comparison:
                rows = [
                    {
                        "Model": name,
                        "Label": r.get("label", "error"),
                        "Confidence": r.get("confidence", 0),
                    }
                    for name, r in comparison.items()
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
