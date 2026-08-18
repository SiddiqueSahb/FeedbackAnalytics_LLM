"""
FeedbackIQ backend — FastAPI application entrypoint.

Run locally:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

The routers underneath (sentiment, search, rag, analytics) each wrap the
existing nlp/, rag/ and evaluate/ modules — nothing in those modules was
changed to build this API. Model files and the FAISS index are loaded lazily
by the modules that need them, on first request, not at startup, so the
server still starts even if a heavy model file is missing; only the
endpoint that needs it will fail.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.deps import require_api_key
from backend.api.routes import analytics, evaluation, rag, search, sentiment
from backend.models.schemas import HealthResponse
from backend.services.analytics_service import warm_cache
from config import settings
from logger import get_logger

log = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm the analytics cache while the server comes up.

    The 407MB parquet is otherwise read during the first dashboard request,
    which puts it in front of the user rather than in front of the deploy.
    Started as a background task rather than awaited, deliberately: uvicorn
    won't accept connections until the startup hook returns, and blocking
    there would delay readiness for every endpoint and risk tripping the
    container health check or a Cloud Run startup probe. This way /api/health
    answers immediately, the read overlaps with startup, and the first
    request either finds the data already resident or waits on the same lock
    the loader has always used.
    """

    task = asyncio.create_task(asyncio.to_thread(warm_cache))

    yield

    if not task.done():
        task.cancel()


app = FastAPI(
    lifespan=lifespan,
    title="FeedbackIQ API",
    description=(
        "Backend for FeedbackIQ — sentiment analysis, complaint categorisation, "
        "semantic search and retrieval-augmented question answering over "
        "customer feedback."
    ),
    version="1.0.0",
)

# The Streamlit frontend runs as a separate process/container and calls this
# API over HTTP, so it needs CORS enabled.
#
# Origins come from ALLOWED_ORIGINS (comma-separated) and default to "*",
# which keeps local development and docker-compose working exactly as before.
# In a deployed environment set it to the frontend's real URL, e.g.
#   ALLOWED_ORIGINS=https://feedbackiq-frontend-xxxx.a.run.app
# so the browser won't let arbitrary sites call the API on a user's behalf.
#
# allow_credentials is deliberately NOT set — combining it with
# allow_origins=["*"] is invalid per the CORS spec (browsers reject
# Access-Control-Allow-Origin: * on credentialed requests). Nothing here
# uses cookie/credential-based auth today, so there's nothing to gain from
# it and it would silently break any future browser client that tries to
# call this API with credentials included.
allowed_origins = [
    origin.strip()
    for origin in settings.ALLOWED_ORIGINS.split(",")
    if origin.strip()
] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with its status code and how long it took."""

    start = time.time()
    response = await call_next(request)
    duration_ms = round((time.time() - start) * 1000, 1)

    log.info(
        "%s %s -> %d (%sms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch anything that slips past route-level error handling."""

    log.exception("Unhandled error on %s %s", request.method, request.url.path)

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )


# Routers. Each route file already defines its own paths ("/predict",
# "/analyse", ...); the prefix here is what makes the frontend's
# "/api/<area>/<path>" URLs match.
#
# The API key dependency is attached here rather than on individual routes,
# so a new endpoint added to any of these routers is protected by default
# instead of being protected only if its author remembered to ask for it.
# "/" and "/api/health" below are deliberately left open — see backend/api/deps.py.
protected = [Depends(require_api_key)]

app.include_router(sentiment.router, prefix="/api/sentiment", dependencies=protected)
app.include_router(search.router, prefix="/api/search", dependencies=protected)
app.include_router(rag.router, prefix="/api/rag", dependencies=protected)
app.include_router(analytics.router, prefix="/api/analytics", dependencies=protected)
app.include_router(evaluation.router, prefix="/api/evaluation", dependencies=protected)


@app.get("/", tags=["Health"], summary="API root")
async def root() -> dict:
    return {"service": "FeedbackIQ API", "status": "running", "docs": "/docs"}


@app.get("/api/health", response_model=HealthResponse, tags=["Health"], summary="Health check")
async def health() -> HealthResponse:
    """
    Reports whether the pieces the frontend depends on are actually present,
    so a 200 from this endpoint means more than "the process is alive".
    """

    data_loaded = os.path.exists(settings.DATA_PATH)
    index_ready = os.path.exists(settings.INDEX_PATH) or os.path.exists(settings.LANGCHAIN_INDEX_PATH)
    llm_ready = bool(settings.GROQ_API_KEY)

    return HealthResponse(
        status="ok",
        environment=os.getenv("ENVIRONMENT", "development"),
        data_loaded=data_loaded,
        index_ready=index_ready,
        llm_ready=llm_ready,
    )
