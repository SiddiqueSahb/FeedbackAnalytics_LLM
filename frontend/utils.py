"""
Shared helpers for calling the FastAPI backend from Streamlit pages.

Every page needs the same three things: the backend base URL, a GET
helper and a POST helper with sensible error handling. Keeping them here
means each page just calls api_get()/api_post() instead of repeating the
same requests.get(...)/try/except boilerplate five times.
"""

import os
import re
import sys

import requests
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import settings

API = os.getenv("API_URL", settings.API_URL)

# Every /api/* route except /api/health now requires this header (see
# backend/api/deps.py). Read from the environment first so the deployed
# frontend can be given the real key without it living in source.
API_KEY = os.getenv("API_KEY", settings.API_KEY)
HEADERS = {"x-api-key": API_KEY}

# The backend loads its FAISS index and models on the first request that
# needs them, not at startup — deliberately, so the server comes up even
# when a model file is missing. The cost is that the first call after a
# cold start reads well over a gigabyte from disk and can take the better
# part of a minute. The old 15s GET timeout was comfortable locally, where
# the index was usually already warm, and failed on a freshly started
# container against an otherwise healthy backend. 90s covers the cold path;
# a genuinely stuck backend still surfaces, just later.
GET_TIMEOUT = int(os.getenv("API_GET_TIMEOUT", "90"))

# Health is a small existence check with no model loading behind it, so it
# should answer fast or be treated as down. Kept short on purpose — the
# sidebar status widget calls it on every page render.
HEALTH_TIMEOUT = int(os.getenv("API_HEALTH_TIMEOUT", "10"))

# Used everywhere a sentiment class is shown — charts, badges, tables —
# so "positive" is always the same green on every page.
SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#F39C12",
}


@st.cache_data(show_spinner=False, ttl=60)
def api_get(
    endpoint: str,
    params: dict | None = None,
    timeout: int | None = None,
) -> dict | list:
    """GET from the backend. Cached for a minute since dashboard data doesn't
    change second to second. Returns {} on any failure."""

    try:
        response = requests.get(
            f"{API}{endpoint}",
            params=params,
            headers=HEADERS,
            timeout=timeout or GET_TIMEOUT,
        )

        if response.ok:
            return response.json()

        if response.status_code in (401, 403):
            st.error(
                "The backend rejected this request's API key. Set API_KEY to the "
                "same value the backend is using (`.env` locally, or the "
                "environment/secret on the deployed frontend)."
            )
            return {}

        st.error(f"API error ({response.status_code}): {response.text[:200]}")

    except requests.exceptions.Timeout:
        st.error(
            "The backend didn't respond in time. If it was just started it may "
            "still be loading the search index — wait a moment and retry."
        )

    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the FeedbackIQ API. Make sure the backend is running "
            f"at {API}."
        )

    except Exception as exc:
        st.error(f"Unexpected error calling the API: {exc}")

    return {}


def api_post(endpoint: str, payload: dict, timeout: int = 60) -> dict | list | None:
    """POST to the backend. Not cached — these are user-triggered actions
    (analyse a review, run a search, ask a question), not passive reads.
    Returns None on failure so callers can tell "no result" apart from
    "empty result"."""

    try:
        response = requests.post(
            f"{API}{endpoint}",
            json=payload,
            headers=HEADERS,
            timeout=timeout,
        )

        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

        if response.status_code in (401, 403):
            st.error(
                "The backend rejected this request's API key. Set API_KEY to the "
                "same value the backend is using."
            )
            return None

        st.error(f"Request failed ({response.status_code}): {detail}")

    except requests.exceptions.ConnectionError:
        st.error(
            "Can't reach the FeedbackIQ API. Make sure the backend is running "
            f"at {API}."
        )

    except requests.exceptions.Timeout:
        st.error("The request took too long and timed out. Please try again.")

    except Exception as exc:
        st.error(f"Unexpected error calling the API: {exc}")

    return None


# Which research question each page demonstrates. Kept here rather than in the
# pages so the mapping is stated once and every page shows it the same way —
# the landing page is organised around these four questions, and a page that
# didn't name its own would break that thread.
RESEARCH_QUESTIONS = {
    "RQ1": "How does sentiment classification performance differ between lexicon, "
           "classical, pretrained transformer and fine-tuned transformer approaches?",
    "RQ2": "How effectively does retrieval-augmented generation improve complaint "
           "exploration compared with a standalone large language model?",
    "RQ3": "Can BERTopic produce a coherent cross-domain complaint taxonomy from "
           "heterogeneous customer reviews?",
    "RQ4": "To what extent can retrieval-augmented answers be traced to, and "
           "verified against, the source reviews used to produce them?",
}


def page_header(title: str, subtitle: str, rq: str | list[str] | None = None) -> None:
    """
    One header treatment for every page: title, one-line explanation, and the
    research question the page speaks to.

    Written as a helper rather than repeated per page because five pages
    drifting into five slightly different header styles is exactly how an app
    stops looking deliberate.
    """
    # Styled header rather than st.title, so pages match the landing page and
    # scale down cleanly on a narrow screen (see frontend/theme.py). The
    # subtitle is written as markdown by callers, so convert the inline
    # emphasis by hand — raw HTML blocks don't get markdown processing.
    subtitle = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", subtitle)
    subtitle = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", subtitle)
    subtitle = re.sub(r"`(.+?)`", r"<code>\1</code>", subtitle)

    st.markdown(
        f'''<div class="fiq-hero">
              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>''',
        unsafe_allow_html=True,
    )

    keys = [rq] if isinstance(rq, str) else (rq or [])
    if keys:
        labels = " · ".join(keys)
        with st.expander(f"Research question: {labels}", expanded=False):
            for key in keys:
                st.markdown(f"**{key}** — {RESEARCH_QUESTIONS.get(key, '')}")

    st.divider()


def render_sources(sources: list[dict], label: str | None = None) -> None:
    """
    Show the reviews an answer was built from.

    This is the visible half of RQ4: an answer that cannot be traced back to
    identifiable evidence is not verifiable, whatever its faithfulness score
    says. Defined once because the chatbot renders it in two places (replayed
    history and the live turn) and the two had drifted apart.
    """
    if not sources:
        return

    n = len(sources)
    with st.expander(label or f"{n} source review{'s' if n != 1 else ''} — the evidence for this answer"):
        for i, source in enumerate(sources, start=1):
            score = source.get("similarity_score")
            bits = [
                f"**{source.get('platform', 'unknown')}**",
                f"★ {source.get('rating', 'N/A')}",
                str(source.get("sentiment_label", "unknown")),
            ]
            if score is not None:
                bits.append(f"similarity {score:.2f}")
            if source.get("review_id"):
                bits.append(f"`{source['review_id']}`")

            st.markdown(f"{i}. " + " · ".join(bits))
            st.markdown(f"> {source.get('text', '')}")
            if i < n:
                st.divider()


def sidebar_status() -> None:
    """
    Small "is the backend actually working" widget, shown in the sidebar on
    every page. Calls /api/health rather than assuming — a green dot here
    only means the process is running; this checks the data, index and LLM
    key are all present too, since any of those missing is the real cause
    behind "why isn't this page showing anything".
    """

    with st.sidebar:

        st.markdown("### FeedbackIQ")

        health = api_get("/api/health", timeout=HEALTH_TIMEOUT)

        if not health:
            st.error("Backend unreachable")
            st.caption(f"Expected at {API}")
            st.divider()
            return

        checks = {
            "Review data": health.get("data_loaded", False),
            "Search index": health.get("index_ready", False),
            "LLM (Groq)": health.get("llm_ready", False),
        }

        # A real status use of colour (genuine system state), not a
        # decorative box — reserving st.success/st.error for cases like
        # this is what makes them still mean something when they appear.
        if all(checks.values()):
            st.success("All systems ready", icon="✅")
        else:
            st.warning("Some components unavailable", icon="⚠️")

        for label, ok in checks.items():
            st.caption(f"{'✅' if ok else '⚠️'} {label}")

        st.divider()
