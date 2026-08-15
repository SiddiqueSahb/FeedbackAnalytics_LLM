"""
Upload — score a batch of reviews from a CSV file using a single sentiment
model. Useful for quickly checking sentiment on a small exported sample
without pasting reviews in one at a time.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import api_get, api_post, sidebar_status

st.set_page_config(
    page_title="Upload | FeedbackIQ",
    page_icon="📤",
    layout="wide",
)

sidebar_status()

MAX_ROWS = 200

st.title("Batch Review Upload")
st.write(
    f"Upload a CSV file with a column of review text to score sentiment for "
    f"many reviews at once. Capped at {MAX_ROWS} rows per upload to keep "
    "response times reasonable."
)

st.caption(
    "This scores sentiment only (no categorisation, search or LLM analysis) — "
    "for the full pipeline on a single review, use the Analyse page."
)

st.divider()

model_options = api_get("/api/sentiment/models")

if not isinstance(model_options, list) or not model_options:
    model_options = [{"id": "distilbert", "name": "DistilBERT (fine-tuned)", "tier": ""}]

model_labels = {m["id"]: f"{m['name']} — {m['tier']}" for m in model_options}

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

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

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read that file as a CSV: {exc}")
        df = None

    if df is not None:

        if df.empty:
            st.warning("The uploaded file has no rows.")

        else:
            text_column = st.selectbox("Which column contains the review text?", df.columns)

            if len(df) > MAX_ROWS:
                st.warning(f"File has {len(df)} rows — only the first {MAX_ROWS} will be scored.")
                df = df.head(MAX_ROWS)

            if st.button("Score Reviews", type="primary"):

                texts = df[text_column].dropna().astype(str).tolist()

                if not texts:
                    st.warning("No usable text found in that column.")

                else:
                    with st.spinner(f"Scoring {len(texts)} reviews..."):
                        results = api_post(
                            "/api/sentiment/batch-predict",
                            {"texts": texts, "model": model_id},
                            timeout=120,
                        )

                    if results:

                        df_results = pd.DataFrame(results)

                        st.success(f"Scored {len(df_results)} reviews.")

                        col1, col2, col3 = st.columns(3)

                        counts = df_results["label"].value_counts()

                        with col1, st.container(border=True):
                            st.metric("Positive", int(counts.get("positive", 0)))
                        with col2, st.container(border=True):
                            st.metric("Negative", int(counts.get("negative", 0)))
                        with col3, st.container(border=True):
                            st.metric("Neutral", int(counts.get("neutral", 0)))

                        st.dataframe(df_results, use_container_width=True, hide_index=True)

                        st.download_button(
                            "Download results as CSV",
                            df_results.to_csv(index=False).encode("utf-8"),
                            file_name="feedbackiq_sentiment_results.csv",
                            mime="text/csv",
                        )
