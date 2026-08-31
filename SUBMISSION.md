# Submission package — contents, exclusions and reproduction

**Author:** Mohammad Asim Siddique (URN 6946816)
**Programme:** MSc Data Science, School of Computer Science and Electronic Engineering, University of Surrey
**Supervisor:** Dr. Santanu Dash
**Dissertation:** *Intelligent Customer Feedback Analytics using Large Language Models: A Stage-Wise Empirical Comparison of Classical Machine Learning, Transformer and Retrieval-Augmented Approaches*

This file explains what is inside the archive, what was left out and why, and how to
regenerate anything that was left out. Appendix B.1 of the dissertation states that all
code, evaluation scripts and result files are provided with the submission; they are.

---

## 1. What is included

| Path | Contents |
|---|---|
| `6946816_rewritten.docx` | The dissertation. |
| `README.md`, `RESULTS.md` | Project overview and the full results write-up. |
| `nlp/`, `rag/`, `backend/`, `frontend/` | Source for the three pipeline stages and the application. |
| `scripts/`, `evaluate/` | Every script that produces a reported figure or table. |
| `tests/`, `test_pipeline.py` | Unit tests covering topic-merging and category-shortlisting logic. |
| `notebooks/` | Preprocessing and EDA, classical models, DistilBERT fine-tuning. |
| `data/results/` | Every result file cited in Chapter 4 and Appendix B (79 files). |
| `data/processed/` | The unified corpus (`reviews_unified.parquet`, 642,692 rows), the BERTopic topic-info files, the merged taxonomy, and the embedding caches used in discovery. |
| `models/distilbert-finetuned-final/` | The fine-tuned DistilBERT classifier. |
| `models/classical/` | Fitted Naive Bayes and logistic regression models with their TF-IDF vectorisers. |
| `mlruns/` | MLflow tracking records for the experiments. |
| `docs/figures/` | Architecture and results figures. |
| `requirements.txt` | Pinned dependency versions. |
| `.env.example` | Template for the one required environment variable. |
| `DEPLOY.md`, `GCP_DEPLOY_RUNBOOK.md`, `docker-compose.yml` | Deployment documentation. |

Random seeds are fixed at 42 throughout.

---

## 2. What is excluded, and how to regenerate it

These were removed to keep the archive to a reasonable size. Each is reproducible from
what is included.

| Excluded | Size | How to regenerate |
|---|---|---|
| `data/raw/` | 5.1 GB | The three source datasets are public — see Section 3.2 of the dissertation. Then run `python scripts/preprocess.py` to rebuild `data/processed/reviews_unified.parquet`. |
| `data/embeddings/` | 3.3 GB | FAISS index and review embedding matrix. Rebuild with `python scripts/build_index.py`, which reads the included parquet file. |
| `models/bertopic_model_negative/`, `_neutral/`, `_positive/` | 3.9 GB | Fitted BERTopic models. Rebuild with `python scripts/discover_categories.py`. The topic-info JSON and merged taxonomy they produced are included in `data/processed/`, so the reported results can be checked without refitting. |
| `venv/` | 2.3 GB | `python -m venv venv && pip install -r requirements.txt` |
| `.git/` | 52 MB | Version history only. |
| `.env` | — | Contains a private API key and is therefore not distributed. Copy `.env.example` to `.env` and supply your own Groq API key. |
| `__pycache__/`, `logs/`, `build.log`, `.DS_Store` | — | Build artefacts. |

---

## 3. Running the code

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your own GROQ_API_KEY
```

The results in Chapter 4 can be reproduced without any raw data, since the unified
corpus and the fitted classifiers are both included:

```bash
python scripts/evaluate_models.py                   # Table 4.3, five-model comparison
python evaluate/evaluate_sentiment_diagnostics.py   # Tables 4.4, 4.5 and the threshold sweep
python evaluate/evaluate_category_discovery.py      # Table 4.6, taxonomy evaluation
python evaluate/generate_all_metrics.py             # per-class metrics and significance tests
python evaluate/generate_figures.py                 # Figures 4.1 to 4.4
```

The retrieval and generation stages need the FAISS index and a Groq API key:

```bash
python scripts/build_index.py                       # rebuilds data/embeddings/
python scripts/compare_embedding.py                 # Table 4.7, retrieval comparison
python evaluate/evaluate_llm_vs_rag.py              # Tables 4.8 and 4.9, RAG vs standalone
python evaluate/ablation_metadata_tagging.py        # Table 4.10, metadata-tagging ablation
```

Tests:

```bash
pytest tests/
```

---

## 4. Ethics

All three datasets are public and were collected by their original publishers. No new
data was gathered and no participants were involved. Reviews are referenced by
identifier only and reviewer names are not used. See Section 3.7 and Appendix A.
