"""
AI Assistant — conversational exploration of customer complaints using
retrieval-augmented generation (RAG). Answers are grounded in retrieved
reviews, with sources shown underneath each answer.
"""

import os
import sys

import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from utils import api_post, sidebar_status

st.set_page_config(
    page_title="Assistant | FeedbackIQ",
    page_icon="💬",
    layout="wide",
)

sidebar_status()

st.title("AI Assistant")
st.write(
    "Ask questions about customer complaint patterns. Answers are generated "
    "only from retrieved reviews, and every answer lists the reviews it drew on."
)

st.info(
    "Try: *\"What are the top complaints on Amazon?\"* or "
    "*\"Why are airline customers most dissatisfied?\"*"
)

st.divider()

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Replay the conversation so far
for message in st.session_state.chat_messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant" and not message.get("grounded", True):
            st.warning(message["content"], icon="⚠️")
        else:
            st.write(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander(f"📚 {len(message['sources'])} source review(s)"):
                for source in message["sources"]:
                    score = source.get("similarity_score")
                    score_text = f" · Similarity: {score:.2f}" if score is not None else ""
                    st.markdown(
                        f"**{source.get('platform', 'unknown')}** · "
                        f"⭐ {source.get('rating', 'N/A')} · "
                        f"{source.get('sentiment_label', 'unknown')}"
                        f"{score_text}\n\n"
                        f"> {source.get('text', '')}"
                    )
                    st.markdown("---")

question = st.chat_input("Ask about customer complaints...")

if question:

    st.session_state.chat_messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    # Backend expects chat_history as prior turns only, in API role/content form
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages[:-1]
    ]

    with st.chat_message("assistant"):

        with st.spinner("Retrieving relevant reviews and generating an answer..."):
            response = api_post(
                "/api/rag/chat",
                {"question": question, "chat_history": history},
                timeout=60,
            )

        if response:
            answer = response.get("answer", "Sorry, I couldn't generate an answer.")
            sources = response.get("sources", [])
            grounded = response.get("grounded", True)

            if not grounded:
                st.warning(answer, icon="⚠️")
            else:
                st.write(answer)

            if sources:
                with st.expander(f"📚 {len(sources)} source review(s)"):
                    for source in sources:
                        score = source.get("similarity_score")
                        score_text = f" · Similarity: {score:.2f}" if score is not None else ""
                        st.markdown(
                            f"**{source.get('platform', 'unknown')}** · "
                            f"⭐ {source.get('rating', 'N/A')} · "
                            f"{source.get('sentiment_label', 'unknown')}"
                            f"{score_text}\n\n"
                            f"> {source.get('text', '')}"
                        )
                        st.markdown("---")

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer, "sources": sources, "grounded": grounded}
            )

        else:
            error_msg = "Sorry, I couldn't reach the backend to answer that."
            st.write(error_msg)
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": error_msg, "sources": [], "grounded": True}
            )

st.divider()

if st.button("Clear conversation"):
    st.session_state.chat_messages = []
    st.rerun()
