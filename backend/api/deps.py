"""
Shared FastAPI dependencies.

Right now this holds one thing: API key authentication.

Why it exists. `settings.API_KEY` was defined in config.py but never read by
anything, which meant every route was open. That's fine on localhost and
actively dangerous the moment the service has a public address — the RAG and
analyse endpoints spend real money on Groq calls per request, so an unguarded
endpoint is an unbounded bill, not just a data-exposure question.

Design notes.

* The key travels in an `x-api-key` header rather than a query parameter, so
  it doesn't end up in access logs, browser history or the Referer header.
* Comparison uses `secrets.compare_digest` rather than `==`. The timing
  difference on a short string is not a practical attack here, but writing
  the naive comparison in an auth path is the kind of thing that gets copied
  into somewhere it does matter.
* The health and root endpoints stay unauthenticated on purpose. Load
  balancers, Cloud Run and Docker's own HEALTHCHECK all probe without
  credentials, and a health check that requires a secret can't do its job.
* If ENVIRONMENT=production and the key is still the development default,
  the process refuses to start. A deployment that silently keeps a key
  published in source control is worse than one that fails loudly on boot —
  same reasoning as the deliberately blank GROQ_API_KEY default in config.py.
"""

from __future__ import annotations

import os
import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config import settings
from logger import get_logger

log = get_logger("api.deps")

API_KEY_HEADER = "x-api-key"

# The value shipped in config.py. Usable locally, never acceptable in a
# deployed environment.
DEV_DEFAULT_KEY = "dev-key-feedbackiq"

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").strip().lower() == "production"

if IS_PRODUCTION and settings.API_KEY in ("", DEV_DEFAULT_KEY):
    raise RuntimeError(
        "ENVIRONMENT=production but API_KEY is unset or still the development "
        f"default ({DEV_DEFAULT_KEY!r}). Set a real API_KEY — via .env locally, "
        "or Secret Manager / --set-secrets in GCP — before starting the server."
    )

if not IS_PRODUCTION and settings.API_KEY == DEV_DEFAULT_KEY:
    log.warning(
        "Using the development API key. Override API_KEY before deploying."
    )

# auto_error=False so a missing header reaches this module and gets an
# explanatory message, rather than FastAPI returning a bare 403.
_api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


async def require_api_key(api_key: str | None = Security(_api_key_header)) -> str:
    """
    Validate the `x-api-key` header. Attached at router level in main.py, so
    individual route functions don't need to mention it.
    """

    if not settings.API_KEY:
        # Not the caller's fault, so this is a 500 rather than a 401.
        log.error("API_KEY is not configured; rejecting authenticated request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: no API key is set.",
        )

    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Send it in the '{API_KEY_HEADER}' header.",
        )

    if not secrets.compare_digest(api_key, settings.API_KEY):
        log.warning("Rejected request with an invalid API key.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
