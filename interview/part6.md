# ROUND 8 — Production, MLOps and Engineering Practice

### Q57. Walk me through your deployment. What breaks first?

"Two Docker containers via compose — FastAPI backend, Streamlit frontend — with data and models mounted as volumes rather than baked into images, because they're multi-GB and change independently of code. Health checks on both, `/api/health` for the backend and Streamlit's `_stcore/health`, wired into compose `depends_on`.

What breaks first is the volume assumption. This design needs a host filesystem, so it works on a VM and doesn't work on Cloud Run, which has no persistent volumes. On Cloud Run I'd fetch the index from GCS on cold start — which adds startup latency proportional to 990MB — or use a service with attached persistent disk.

Second thing that breaks: `lru_cache(maxsize=1)` on the vectorstore loader means each worker process holds its own 990MB copy. Scale to four workers and you're at ~4GB before serving a request."

**Follow-ups.** How do you do zero-downtime deploys with a 990MB artefact? What's your rollback story?

---

### Q58. How do you version models and data?

"Honestly: incompletely, and I'd rather say so than describe an aspiration.

What exists — MLflow tracks parameters, metrics and artefacts per run, so a result traces to its configuration. Dependency versions are pinned. Seeds fixed at 42. Every reported figure is written to disk by the script that computed it.

What doesn't exist — the ~4GB of index and model artefacts aren't in version control, and my split contamination is a direct consequence: the corpus version DistilBERT trained on was overwritten and can't be recovered, so the contamination extent is unquantifiable.

What I'd build: DVC or an artefact registry for data and index versioning, content-hashed so a run records exactly which corpus produced it. Model registry with stage transitions rather than a folder called `distilbert-finetuned-final`. And a CI assertion that training and eval ID sets don't intersect — that single check would have caught the incident that produced my most consequential caveat."

---

### Q59. What's in your CI pipeline, and what should be?

"What's there: GitHub Actions linting backend and frontend, and building both images on push and PR.

What's missing, and this is a real gap: there's no meaningful test suite. My `tests/` directory is mostly interactive print-and-inspect scripts with zero assertions — `test_rag.py` is an input() loop. The one genuine exception is `test_merge_topics.py`, with real assertions on BERTopic cluster-merge behaviour.

What I'd add, in priority order. Unit tests on the pure logic — threshold filtering, MMR intersection, retry classification, label mapping — all of which are testable without models. Contract tests on the API using FastAPI's TestClient with a stubbed retriever. A data-leakage assertion. A small golden-set regression test: five fixed questions with recorded retrieval IDs, failing if retrieval changes unexpectedly. And the guard probe set, so scope-classifier precision is tracked rather than assumed.

I'd flag that calling a directory `tests/` when it contains manual scripts is itself misleading, and I'd rename it."

---

### Q60. Security review of your API. What's wrong?

"Several things, and I'll order them by severity.

The API key is a static value in config with a development default — `dev-key-feedbackiq`. That's a single shared secret with no rotation and no per-client identity. It needs proper auth, ideally OAuth or signed tokens, with per-tenant keys.

No rate limiting. An unauthenticated or authenticated caller can drive unbounded LLM spend. That's a cost-availability attack, and it's the one I'd fix first because it has a direct financial impact.

Prompt injection is the interesting one and partly handled. Retrieved reviews are user-generated content going into a prompt, so a review containing 'ignore previous instructions' is an injection vector. My prompt explicitly instructs the model to treat retrieved reviews as data to summarise rather than instructions to follow — that's rule 2. But a prompt instruction is exactly the kind of soft control I criticised elsewhere. Real mitigation is structural separation of instructions from data, and output validation.

Also: CORS had `allow_credentials` dropped, which is correct for a token-based API. And errors return generic messages to clients with detail logged server-side, so stack traces don't leak."

**Follow-ups.** How would you test for prompt injection? What's your PII position?

**PII answer.** "Reviews are public data, referenced by ID only, and no attempt is made to link them to individuals. But review *text* occasionally contains personal detail the reviewer wrote themselves, and that text goes into prompts sent to a third-party API. For a production system with a DPA that needs addressing — either PII scrubbing before indexing, or a self-hosted generator."

---

### Q61. How would you reduce cost by 50%?

"Four levers, ordered by effort against saving.

Semantic caching is the largest and cheapest win. Analyst questions repeat heavily. Cache on quantised query embedding with an index-version key, and every hit eliminates a generation call entirely — that's the whole per-query cost, not a fraction.

Second, prompt size. Generation dominates latency and cost, and cost scales with input tokens. I cap reviews at 1,000 characters, but I could compress context — extractive summarisation of retrieved reviews before insertion, or fewer reviews if the retrieval-depth ablation showed k=3 performs comparably to k=5. That ablation was designed and never run, which is a concrete cost-saving experiment sitting undone.

Third, the history-aware retriever adds an LLM call per multi-turn query just to rewrite the question. For single-turn queries it's skipped, but for conversations it's a second call. A smaller model, or a heuristic for cases where rewriting isn't needed, halves calls on that path.

Fourth, model tiering — route simple questions to a smaller model and reserve the larger one for complex ones. That needs a classifier, which is its own reliability risk, so I'd do it last."

---

### Q62. Your logging. What would you change?

"Currently structured logging via a `get_logger` wrapper, with the retriever logging threshold decisions and the pipeline logging question, history length, source count and answer length. Errors log with `exc_info=True` so tracebacks are captured.

What's wrong: the log line records *that* a refusal happened but the reason is embedded in which message string got returned, not in a structured field. That's precisely why my scope-guard bug was hard to see — I had to read returned text rather than query a field.

What I'd change: emit structured JSON with a `refusal_reason` enum, a request ID propagated through the whole call chain, retrieval score distribution rather than just the count, and token counts for cost attribution. Then refusal-rate-by-reason becomes a query rather than an investigation."

---

### Q63. How would you handle a 3am page saying answers have become nonsense?

"Triage in order of likelihood and blast radius.

First, is it retrieval or generation? Check whether retrieved review IDs are sane for the query. If retrieval is returning irrelevant documents, suspect the index — wrong version loaded, a partial rebuild, or an embedding-model mismatch between index build and query time. That last one is nasty and silent: if the index was built with MiniLM and queries are encoded with a different model, every similarity is meaningless but nothing errors.

Second, if retrieval looks right, suspect the generator — provider model change, a silent version bump, or a prompt-template regression from a deploy.

Third, check the guards. A threshold change or a scope-classifier regression would show as a refusal-rate shift, which is why I'd want that metric segmented by reason.

Immediate mitigation before root cause: roll back to the previous index artefact and previous deploy. Blue-green artefacts exist precisely so this is a config change rather than a rebuild.

The embedding-mismatch case is the one I'd guard against structurally — record the embedding model name and version in the index metadata and assert it matches at query time, so it fails loudly instead of degrading silently."

---

# HIRING COMMITTEE ASSESSMENT

Honest scoring. Treat the low marks as your prep list.

| Dimension | Score | Reasoning |
|---|---|---|
| Research quality | **8/10** | Genuinely strong. Stage-wise design, per-class and per-platform reporting, significance testing, an executed ablation, honest negative results. The diagnostic chain on RoBERTa is better than most MSc work and better than some published applied papers. |
| ML engineering | **6.5/10** | Sound model choices with real justification. Held back by the split contamination and by an unvalidated threshold in a production path. |
| Evaluation rigour | **7/10** | Excellent on the sentiment stage — 128,539 held out, McNemar, three diagnostics. Weak on generation: n=5, self-grading judge. You know this, which helps. |
| Architecture | **6/10** | Clean separation, sensible guard ordering, thoughtful FAISS wrapping. Single-process, no sharding, in-process index — research-grade, not production-grade. |
| Backend / API | **6/10** | FastAPI with async offloading and Pydantic validation is right. Auth is a static shared secret, no rate limiting. |
| Code quality | **5.5/10** | Readable with unusually good explanatory comments. No real test suite is the problem — `tests/` contains manual scripts with almost no assertions. |
| Deployment | **5.5/10** | Docker split by resource profile is correct thinking. Volume dependency means it doesn't run on serverless as-is. |
| Scalability | **5/10** | Not designed for it, and you'd say so. You can articulate the production shape, which counts for a lot. |
| Documentation | **9/10** | Exceptional. Every figure written by the script that computed it, seeds fixed, versions pinned, limitations disclosed at the point they arise. |
| Innovation | **5/10** | No novel method, and none claimed. The contribution is comparative and diagnostic, which is the right level for MSc. |
| **Overall** | **6.5/10** | |

**Company-by-company read.**

*Google (L4)* — Documentation and rigour play well. Expect heavy pressure on the coding round and on scaling; the L4 bar is production engineering, and this is research code. Practise Q45–Q51 until they're automatic.

*Meta (E4)* — Values shipping and metrics. Lead with the ablation and the A/B design answer. The n=5 evaluation will be attacked hard; the honest framing is your best defence.

*Amazon (Applied Scientist I / SDE II)* — Leadership Principles map well. "Dive Deep" is the RoBERTa diagnosis. "Earn Trust" is the scope-guard self-correction. "Are Right, A Lot" is a risk — prepare to explain the contamination as a process failure with a named control, not an excuse.

*OpenAI / Anthropic (MLE)* — Strongest fit. Grounding, hallucination, refusal behaviour, evaluation honesty are the daily concerns. The scope-guard story and the faithfulness-vs-relevancy divergence are directly relevant. Expect deep questions on judge reliability.

*Databricks / Snowflake* — Will focus on the data pipeline and scale. Q52–Q54 are your weakest area; over-prepare them.

*NVIDIA* — Wants inference optimisation. You have fp16 and distillation reasoning but no profiling, quantisation or kernel-level work. Weakest fit of the set.

*Microsoft / Apple / Netflix / Uber / Stripe* — Standard MLE loops. The project supports the conversation; coding and system design decide the outcome.

---

# PREPARATION PLAN

**Highest leverage, in order.**

1. **Write the guard probe set and calibrate the threshold.** ~100 labelled questions, sweep, report. Converts your weakest answer into a strong one, and it's a few hours of work. If you do one thing, do this.
2. **Run the retrieval-depth ablation.** k ∈ {3,5,10,20}. Your component metrics point straight at retrieval; this closes the biggest evidence gap and gives you a second measured ablation.
3. **Write real tests.** Threshold filtering, MMR intersection, retry classification, a leakage assertion. Turns a 5.5 into a 7 on code quality and it's genuinely quick.
4. **Practise Q45–Q51 cold.** Typed, no autocomplete, out loud. This is where MLE offers are lost.
5. **Rehearse Q52 aloud** until the structure — requirements, estimates, components, failure modes — is automatic.

**Answers to have word-perfect:** Q1, Q5, Q23, Q34, and the split-contamination defence (Attack 1). Those are your five best moments; everything else is supporting.

**Never do these.** Don't quote n=5 figures as if they're precise. Don't defend 0.35 as correct — defend the method for finding it. Don't claim the system is production-ready. Don't hide the contamination; lead with it as a process lesson.
