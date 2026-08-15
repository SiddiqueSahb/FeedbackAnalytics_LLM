# FeedbackIQ backend — FastAPI service.
#
# Build from the project root (data/ and models/ are intentionally left out
# of the image — see .dockerignore — and mounted as volumes at runtime
# instead, since they're multiple GB and change independently of the code):
#
#   docker build -f backend/Dockerfile -t feedbackiq-backend .
#
# Or just: docker compose up backend

FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required at runtime by torch and faiss-cpu (OpenMP). git is
# needed by a couple of pip packages that install directly from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies before copying the rest of the code, so this slow
# layer is only rebuilt when the requirements files actually change.
COPY requirements.txt ./requirements.txt
COPY backend/requirements-extra.txt ./backend/requirements-extra.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt -r backend/requirements-extra.txt \
    && pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Application code. Only the pieces the backend actually imports —
# everything else (data/, models/, notebooks/, docs/, tests/, scripts/,
# evaluate/) is excluded by .dockerignore.
COPY backend/ ./backend/
COPY nlp/ ./nlp/
COPY rag/ ./rag/
COPY config.py logger.py exceptions.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
