# FeedbackIQ frontend — Streamlit UI.
#
# Build from the project root:
#   docker build -f frontend/Dockerfile -t feedbackiq-frontend .
#
# Or just: docker compose up frontend
#
# This image is deliberately light — no torch/faiss/transformers — since
# the frontend only calls the backend over HTTP (see API_URL below).

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt ./frontend/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r frontend/requirements.txt

COPY frontend/ ./frontend/
COPY config.py logger.py ./
COPY .streamlit/ ./.streamlit/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_URL=http://backend:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
