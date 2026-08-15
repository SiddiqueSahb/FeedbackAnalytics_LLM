"""
FeedbackIQ — AI-Powered Customer Feedback Analytics

Run:
    streamlit run frontend/app.py
"""

import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import sidebar_status

# Page Configuration
st.set_page_config(
    page_title="FeedbackIQ",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

sidebar_status()

# Header — plain, left-aligned, theme-driven. No custom CSS/HTML: st.title
# already picks up the navy primaryColor and font from .streamlit/config.toml.
st.title("FeedbackIQ")
st.caption(
    "Customer feedback analytics — sentiment classification, complaint "
    "categorisation and retrieval-augmented question answering across "
    "Amazon, Yelp and Twitter Airline reviews."
)

st.divider()


# Platform Overview — bordered KPI cards instead of bare metrics, so the
# headline numbers read as a stat panel rather than floating text.
st.markdown("### Platform Overview")

col1, col2, col3, col4 = st.columns(4)

with col1, st.container(border=True):
    st.metric("Customer Reviews", "642K+")

with col2, st.container(border=True):
    st.metric("Platforms", "3")

with col3, st.container(border=True):
    st.metric("Complaint Categories", "24")

with col4, st.container(border=True):
    st.metric("Embedding Size", "384")

st.divider()

# Processing pipeline — a plain bordered card with a compact step list
# instead of a large coloured info box with arrow emoji between stages.
st.markdown("### How It Works")

with st.container(border=True):
    st.markdown(
        "**Review**  →  **Sentiment analysis** (fine-tuned DistilBERT)  →  "
        "**Complaint categorisation** (BERTopic + zero-shot classification)  →  "
        "**Semantic search** (Sentence Transformers + FAISS)  →  "
        "**Grounded answer** (LangChain + Groq LLM)"
    )
    st.caption(
        "Each analysed review also produces a summary, keywords, severity, "
        "priority, department routing and an executive summary."
    )

st.divider()

# Features
st.markdown("### Key Features")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("Interactive Dashboard")
        st.write("""
Explore customer feedback through interactive visualisations.

- Sentiment trends
- Complaint categories
- Platform comparison
- Business insights
""")

with col2:
    with st.container(border=True):
        st.subheader("AI Review Analysis")
        st.write("""
Analyse individual customer reviews using AI.

- Sentiment prediction
- Complaint categorisation
- AI-generated summary
- Business recommendations
""")

col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.subheader("Semantic Search")
        st.write("""
Find similar customer reviews based on meaning rather than exact keywords using FAISS-powered semantic search.
""")

with col4:
    with st.container(border=True):
        st.subheader("AI Assistant")
        st.write("""
Ask questions about customer feedback in natural language.

Example questions:

- What are the most common complaints?
- Which products receive the most negative feedback?
- Summarise battery-related issues.
""")

st.divider()

# Technology Stack — neutral bordered card, not a coloured banner.
st.markdown("### Technology Stack")

tech1, tech2 = st.columns(2)

with tech1, st.container(border=True):
    st.markdown("**Machine Learning & NLP**")
    st.markdown("- Fine-tuned DistilBERT\n- BERTopic\n- BART zero-shot classification")
    st.markdown("**Retrieval & LLM**")
    st.markdown(
        "- Sentence Transformers\n- FAISS vector search\n"
        "- Retrieval-augmented generation\n- LangChain\n- Groq LLM"
    )

with tech2, st.container(border=True):
    st.markdown("**Backend & Deployment**")
    st.markdown("- FastAPI\n- Streamlit\n- Docker")
    st.markdown("**Programming**")
    st.markdown("- Python\n- PyTorch\n- Pandas\n- NumPy")

st.divider()

# Business Value
st.markdown("### Business Value")

st.write("""
FeedbackIQ helps organisations understand customer feedback at scale by automatically:

- Detecting customer sentiment
- Identifying complaint categories
- Retrieving similar historical reviews
- Generating actionable business recommendations
- Prioritising critical customer issues
- Routing cases to the appropriate business department
- Producing concise executive summaries for decision-makers
""")

st.divider()

# Navigation
st.markdown("### Explore the Platform")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.page_link("pages/01_Dashboard.py", label="Dashboard", icon="📊", use_container_width=True)

with nav2:
    st.page_link("pages/02_Analyse.py", label="Review Analysis", icon="🧠", use_container_width=True)

with nav3:
    st.page_link("pages/03_Search.py", label="Semantic Search", icon="🔎", use_container_width=True)

with nav4:
    st.page_link("pages/04_Chatbot.py", label="AI Assistant", icon="💬", use_container_width=True)

st.divider()

# Footer
st.caption(
    "**FeedbackIQ v1.0** — AI-Powered Customer Feedback Analytics Platform  \n"
    "Built with Python, FastAPI, Streamlit, Hugging Face, DistilBERT, BERTopic, "
    "Sentence Transformers, FAISS, LangChain and Groq LLM."
)
