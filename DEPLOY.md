# Running FeedbackIQ

## Locally, without Docker

```bash
# Backend (from the project root)
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
streamlit run frontend/app.py
```

The frontend reads `API_URL` from `config.py` / the environment, defaulting to `http://localhost:8000`.

## With Docker

```bash
docker compose up --build
```

- Backend: http://localhost:8000 (interactive docs at `/docs`)
- Frontend: http://localhost:8501

`data/`, `models/`, `mlruns/` and `logs/` are mounted from the host, not copied into the images — they're multi-GB and already produced by the existing scripts, so there's no reason to bake them into a container image. Make sure they exist locally before starting (i.e. you've already run the preprocessing/training/index-building scripts).

The backend needs a Groq API key to answer chat questions and generate business summaries. It's already defaulted in `config.py`; to use your own, put `GROQ_API_KEY=...` in `.env` at the project root — `docker-compose.yml` loads that file automatically.

## What was added

- `backend/` — FastAPI app (`main.py`) with four route groups: sentiment, search, rag, analytics. Each route is a thin layer over the existing `nlp/` and `rag/` modules — no changes were made to those.
- `frontend/` — Streamlit pages: Dashboard (already existed), Analyse, Search, Chatbot, Upload.
- `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml` — images built from the project root; heavy data/model directories excluded via `.dockerignore` and mounted as volumes instead.
- `.github/workflows/ci.yml` — lints `backend/` and `frontend/` and builds both images on every push/PR.

## Where the roadmap items plug in

Nothing below is built yet — this is just where each piece would go, based on how the project is currently structured:

- **MongoDB**: swap `_load_df()` in `backend/services/analytics_service.py` for a Mongo query, add a `mongo` service to `docker-compose.yml`.
- **Live ingestion (Amazon/Yelp/Twitter APIs)**: a new `backend/services/ingestion_service.py` writing into Mongo instead of the parquet file, called on a schedule or webhook.
- **Retraining pipeline**: a script under `scripts/` that reruns `train_classical_models.py`/the fine-tuning notebook when new labelled data arrives, logged to the MLflow instance already configured in `config.py`.
- **DVC**: `dvc init`, then track `data/` and `models/` with it instead of (or alongside) the current `.gitignore` exclusions.
- **Prometheus/Grafana**: add a `/metrics` endpoint to `backend/main.py` (e.g. via `prometheus-fastapi-instrumentator`) and `prometheus`/`grafana` services to `docker-compose.yml`.
