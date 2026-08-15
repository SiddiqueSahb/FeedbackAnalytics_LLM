"""
Shared helpers for calling the FastAPI backend from Streamlit pages.

Every page needs the same three things: the backend base URL, a GET
helper and a POST helper with sensible error handling. Keeping them here
means each page just calls api_get()/api_post() instead of repeating the
same requests.get(...)/try/except boilerplate five times.
"""

import os
import sys

import requests
import streamlit as st

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import settings

API = os.getenv("API_URL", settings.API_URL)

# Used everywhere a sentiment class is shown — charts, badges, tables —
# so "positive" is always the same green on every page.
SENTIMENT_COLORS = {
    "positive": "#2ECC71",
    "negative": "#E74C3C",
    "neutral": "#F39C12",
}


@st.cache_data(show_spinner=False, ttl=60)
def api_get(endpoint: str, params: dict | None = None) -> dict | list:
    """GET from the backend. Cached for a minute since dashboard data doesn't
    change second to second. Returns {} on any failure."""

    try:
        response = requests.get(f"{API}{endpoint}", params=params, timeout=15)

        if response.ok:
            return response.json()

        st.error(f"API error ({response.status_code}): {response.text[:200]}")

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
        response = requests.post(f"{API}{endpoint}", json=payload, timeout=timeout)

        if response.ok:
            return response.json()

        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text

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

        health = api_get("/api/health")

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
