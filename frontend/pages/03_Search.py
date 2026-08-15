"""
Semantic Search — find reviews similar in meaning to a natural language
query, using the FAISS index rather than exact keyword matching.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import api_post, sidebar_status

st.set_page_config(
    page_title="Search | FeedbackIQ",
    page_icon="🔎",
    layout="wide",
)

sidebar_status()

st.title("Semantic Search")
st.write(
    "Search customer reviews by meaning, not exact words. For example, "
    "\"phone won't turn on\" will also match reviews saying \"device is dead\"."
)

st.divider()

query = st.text_input(
    "Search query",
    placeholder="e.g. battery draining too fast",
)

st.sidebar.header("Filters")

platform_filter = st.sidebar.selectbox(
    "Platform", ["Any", "amazon", "yelp", "twitter_airline"]
)

sentiment_filter = st.sidebar.selectbox(
    "Sentiment", ["Any", "positive", "negative", "neutral"]
)

min_rating = st.sidebar.slider("Minimum rating", 1.0, 5.0, 1.0, 0.5)

top_k = st.sidebar.slider("Number of results", 5, 50, 10)

search_clicked = st.button("Search", type="primary")

st.divider()

if search_clicked:

    if len(query.strip()) < 2:
        st.warning("Please enter a search query of at least 2 characters.")

    else:
        with st.spinner("Searching..."):
            results = api_post(
                "/api/search",
                {
                    "query": query,
                    "top_k": top_k,
                    "platform_filter": None if platform_filter == "Any" else platform_filter,
                    "sentiment_filter": None if sentiment_filter == "Any" else sentiment_filter,
                    "min_rating": min_rating,
                },
            )

        if results:

            st.success(f"Found {len(results)} matching reviews.")

            df = pd.DataFrame(results)

            df = df.rename(columns={
                "text": "Review",
                "platform": "Platform",
                "rating": "Rating",
                "sentiment_label": "Sentiment",
                "similarity_score": "Similarity",
            })

            st.dataframe(
                df[["Review", "Platform", "Rating", "Sentiment", "Similarity"]],
                use_container_width=True,
                hide_index=True,
            )

        elif results is not None:
            st.info("No reviews matched your search and filters.")

st.divider()
st.caption(
    "Powered by SentenceTransformer embeddings (all-MiniLM-L6-v2) and a FAISS "
    "vector index built over the full review corpus."
)
