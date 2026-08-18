# ROUND 6 — Coding Questions From Your Own Project

Expect to write these on a whiteboard or shared editor with no autocomplete. Practise typing them cold.

### Q45. Implement cosine similarity and top-k retrieval from scratch. No FAISS, no sklearn.

```python
import numpy as np

def normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows. Guard against zero vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0                      # avoid divide-by-zero
    return matrix / norms

def top_k_cosine(query: np.ndarray,
                 corpus: np.ndarray,
                 k: int = 5,
                 threshold: float | None = None):
    """Return (index, score) pairs, highest first.

    corpus: (N, d) assumed pre-normalised — normalise once at index time,
    not per query, or you pay O(N*d) needlessly on every request.
    """
    q = query / (np.linalg.norm(query) or 1.0)
    scores = corpus @ q                          # (N,) cosine, since both unit norm

    if threshold is not None:
        keep = np.where(scores >= threshold)[0]
        if keep.size == 0:
            return []                            # caller must treat as "no evidence"
        scores_kept = scores[keep]
        k = min(k, keep.size)
        # argpartition is O(N) vs O(N log N) for a full sort
        part = np.argpartition(-scores_kept, k - 1)[:k]
        order = part[np.argsort(-scores_kept[part])]
        return [(int(keep[i]), float(scores_kept[i])) for i in order]

    k = min(k, scores.shape[0])
    part = np.argpartition(-scores, k - 1)[:k]
    order = part[np.argsort(-scores[part])]
    return [(int(i), float(scores[i])) for i in order]
```

**Points the interviewer is looking for.** Normalising the corpus once at index time, not per query. `argpartition` rather than a full sort — O(N) versus O(N log N). Zero-vector guard. Returning empty rather than raising when nothing clears the threshold, because that's a valid state your pipeline must handle.

**Follow-up: make it batched.**

```python
def top_k_cosine_batch(queries: np.ndarray, corpus: np.ndarray, k: int = 5):
    """queries: (B, d) -> (B, k) indices and scores. One matmul, not B."""
    Q = normalise(queries)
    scores = Q @ corpus.T                        # (B, N)
    k = min(k, corpus.shape[0])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[rows, part], axis=1)
    idx = part[rows, order]
    return idx, scores[rows, idx]
```

Say out loud why this is faster: one BLAS-backed matmul rather than B separate ones, so you get cache locality and multithreading for free.

---

### Q46. Implement MMR selection.

```python
def mmr(query: np.ndarray,
        candidates: np.ndarray,
        cand_ids: list,
        k: int = 5,
        lambda_mult: float = 0.5):
    """Maximal Marginal Relevance.

    Greedily pick the candidate maximising
        lambda * sim(q, d) - (1 - lambda) * max_{s in selected} sim(d, s)
    """
    q = query / (np.linalg.norm(query) or 1.0)
    C = normalise(candidates)
    rel = C @ q                                  # relevance to query
    sim = C @ C.T                                # pairwise candidate similarity

    selected, remaining = [], list(range(len(cand_ids)))
    first = int(np.argmax(rel))
    selected.append(first); remaining.remove(first)

    while len(selected) < min(k, len(cand_ids)):
        best, best_score = None, -np.inf
        for i in remaining:
            redundancy = sim[i, selected].max()
            score = lambda_mult * rel[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best, best_score = i, score
        selected.append(best); remaining.remove(best)

    return [cand_ids[i] for i in selected]
```

**Complexity:** O(k · |C|) similarity lookups after an O(|C|²·d) pairwise matrix. That's why MMR runs over a small candidate pool (fetch_k=20) rather than the whole corpus.

**Follow-up you should pre-empt:** "In my pipeline I then intersect this with the set that cleared the threshold, because MMR only controls diversity — it has no notion of whether anything is relevant."

---

### Q47. Implement an LRU cache. Then tell me the eviction cost.

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)                 # O(1)
        return self._d[key]

    def put(self, key, value):
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = value
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)          # evict least-recently-used, O(1)
```

**Answer on cost:** all operations O(1). `OrderedDict` is a hash map plus a doubly-linked list — the map gives O(1) lookup, the list gives O(1) reordering and eviction. If asked to implement without `OrderedDict`, write the dict-plus-linked-list by hand; that's the actual question.

**Project-specific follow-up: what would you cache here, and what's the invalidation key?**

"Not raw question strings — 'battery complaints' and 'complaints about battery' should hit the same entry. I'd cache on the *embedded* query, quantised or bucketed, with a similarity threshold for a cache hit. That's a semantic cache. Invalidation key has to include the index version, because a cached answer references retrieved review IDs and those are only valid for the index that produced them. Cache key = hash(query embedding bucket, index version, k, threshold)."

---

### Q48. Write an async FastAPI endpoint that doesn't block the event loop.

```python
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    chat_history: list[dict] = Field(default_factory=list)

class Source(BaseModel):
    review_id: str
    platform: str
    similarity_score: float | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    retrieval_count: int

@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    try:
        # ask() is synchronous and CPU/IO-bound (FAISS + external LLM).
        # to_thread moves it off the event loop so concurrent requests
        # are not serialised behind it.
        result = await asyncio.to_thread(ask, req.question, req.chat_history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Index unavailable: {e}")
    except Exception:
        # never leak internals to the client; log server-side with traceback
        log.exception("ask failed for question=%r", req.question[:80])
        raise HTTPException(status_code=500, detail="Internal error")
    return AskResponse(**result)
```

**What they're checking.** That you know `async def` around synchronous work buys nothing. Pydantic validation with bounds. Distinguishing a 503 (dependency missing) from a 500 (unexpected). Not leaking stack traces to clients. Logging with the exception attached.

**Follow-up: how would you add a timeout?** — `asyncio.wait_for(asyncio.to_thread(...), timeout=30)` and return 504. Note that `to_thread` can't actually be cancelled, so the thread keeps running; a hard bound needs a process pool or a timeout inside the LLM client itself, which is why I set `timeout=30` on the ChatGroq client.

---

### Q49. Add retry-with-backoff for a rate-limited API. Retry only what should be retried.

```python
import random, re, time

RETRYABLE_STATUS = {429, 500, 502, 503, 504}

def _retry_after(exc: Exception, default: float) -> float:
    """Providers often state the wait in the error body."""
    m = re.search(r"try again in ([\d.]+)s", str(exc))
    if m:
        try:
            return float(m.group(1)) + 0.5
        except ValueError:
            pass
    return default

def call_with_retry(fn, *args, max_retries: int = 2, base: float = 2.0, **kwargs):
    """Retry transient failures only. Non-retryable errors raise immediately."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            transient = ("rate_limit" in msg
                         or any(str(c) in msg for c in RETRYABLE_STATUS))
            if not transient or attempt >= max_retries:
                raise
            wait = _retry_after(exc, base * (2 ** attempt))
            wait += random.uniform(0, wait * 0.1)     # jitter: avoid thundering herd
            log.warning("transient failure, retrying in %.1fs (%d/%d)",
                        wait, attempt + 1, max_retries)
            time.sleep(wait)
```

**Points to make unprompted.** Exponential backoff with jitter, so simultaneous clients don't retry in lockstep. Honouring a server-supplied `Retry-After` over your own schedule. Retrying *only* transient classes — a bad API key retried three times is three failures and a slower error. And the retry budget differs by context: my offline evaluation script uses 6 retries at 8s, the live request path uses 2 at ~2s, because a person is waiting.

**Follow-up: what about a circuit breaker?** — "Right answer for sustained failure. After N consecutive failures, open the circuit and fail fast for a cooldown window rather than retrying every request. Retry handles blips; circuit breakers handle outages, and conflating them means your latency collapses during a provider incident."

---

### Q50. Given the predictions CSV, compute macro F1 without sklearn.

```python
def macro_f1(y_true: list[str], y_pred: list[str]) -> tuple[float, dict]:
    labels = sorted(set(y_true) | set(y_pred))
    per_class, f1s = {}, []
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "support": tp + fn}
        f1s.append(f1)
    return sum(f1s) / len(f1s), per_class
```

**Say this while writing it:** the zero-denominator guards matter, and not just defensively. VADER's neutral recall is 0.048 — a class it almost never predicts. If a model predicts a class zero times, precision is 0/0, and returning 0.0 rather than raising is the convention. That case is exactly where my central finding lives.

**Follow-up: implement McNemar's exact test.**

```python
from math import comb

def mcnemar_exact(correct_a: list[bool], correct_b: list[bool]) -> tuple[int, int, float]:
    """Two-sided exact binomial test on discordant pairs."""
    b = sum(1 for a, c in zip(correct_a, correct_b) if a and not c)   # A right, B wrong
    c = sum(1 for a, d in zip(correct_a, correct_b) if not a and d)   # B right, A wrong
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)
```

Point out that concordant pairs are discarded entirely — that's the whole idea, and it's why the test needs paired predictions rather than two accuracy numbers.

---

### Q51. Design the retrieval pipeline as a class. Show me your OOP.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    text: str
    score: float
    metadata: dict

class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievedDoc]: ...

class DenseRetriever(Retriever):
    def __init__(self, index, encoder, *, threshold: float = 0.35):
        self._index, self._encoder, self._threshold = index, encoder, threshold

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        vec = self._encoder.encode(query, normalize_embeddings=True)
        hits = self._index.search(vec, k=k)
        return [d for d in hits if d.score >= self._threshold]

class GuardedRetriever(Retriever):
    """Decorator: composes a scope check in front of any retriever."""
    def __init__(self, inner: Retriever, guard):
        self._inner, self._guard = inner, guard

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedDoc]:
        if not self._guard.in_scope(query):
            return []                      # empty means "do not generate"
        return self._inner.retrieve(query, k)
```

**Design points to articulate.** `Retriever` as an ABC so a sparse or hybrid retriever drops in without touching callers. `GuardedRetriever` as a decorator rather than an `if` inside `DenseRetriever` — the guard is orthogonal to retrieval and composes with any implementation. Frozen dataclass so retrieved documents can't be mutated downstream. Threshold injected, not a module constant, so it's testable and configurable per deployment.

**The weakness in my actual code, and admit it:** in `rag/pipeline.py` the threshold is a module-level constant and the guard is inline in `ask()`. It works but it's harder to test and you can't run two configurations side by side — which is exactly what made the ablation awkward to write.

---

# ROUND 7 — System Design

### Q52. Scale this to 100 million reviews, 500 QPS, p99 under 300ms, multi-tenant.

Structure your answer in this order — requirements, then estimates, then components, then failure modes.

**Estimates first.** 100M × 384 dims × 4 bytes ≈ 154GB of raw vectors. That doesn't fit one instance, so sharding is forced rather than optional. At 500 QPS with p99 300ms and generation taking ~1.5s, generation cannot be in the synchronous path for every request — that constraint drives the design.

**Components.**

*Ingestion:* reviews land on a queue (Kafka/PubSub). A stateless embedding service consumes, batches, encodes on GPU, writes vectors plus metadata to object storage. Batching matters — GPU utilisation on single documents is terrible.

*Index build:* offline job producing versioned, immutable shard artefacts. IVF-PQ or HNSW, not Flat — at 100M, exhaustive scan is dead. Shard by tenant if tenants are large, or by hash if many small tenants, because per-tenant shards give isolation and let you delete a tenant cleanly.

*Serving:* stateless retrieval service, N replicas per shard, loading shard artefacts read-only from object storage. A scatter-gather coordinator queries relevant shards and merges top-k. Behind a load balancer.

*Caching:* Redis with two layers — exact-match on normalised query string, and a semantic layer on quantised query embedding. Analyst workloads are repetitive, so hit rate should be meaningful. Cache key must include index version and tenant.

*Generation:* async. Return retrieved evidence synchronously within the 300ms budget, then stream the generated answer over SSE or WebSocket. This is the key move — you meet p99 on retrieval and let generation arrive progressively.

*Multi-tenancy:* tenant ID in every request, enforced at the shard-routing layer, not filtered post-retrieval. Post-filtering is both slower and a data-leak risk if the filter is ever missed.

**Failure modes to raise unprompted.** Shard hot-spotting if one tenant dominates. Index rebuild causing a serving gap — solved with blue-green artefact swap, never in-place mutation. LLM provider outage — circuit breaker, and degrade to returning retrieved reviews without a generated summary, which is still useful. Cold start on replica scale-up loading a multi-GB shard, mitigated by pre-warmed pools.

---

### Q53. How do you add a review without rebuilding the index?

"`IndexFlatL2` supports `add()`, so appends are cheap. Deletes are the problem — Flat has no delete, and IVF's `remove_ids` is expensive and fragments the structure.

Production pattern is tombstoning: mark deleted in a metadata store, filter post-retrieval, over-fetch to compensate for filtered results, and rebuild on a schedule when the tombstone ratio crosses a threshold — maybe 10%.

For a stream of new reviews I'd keep a small hot index of recent additions searched alongside the main index, merging results, then fold it into the main index at each rebuild. That bounds staleness to minutes without paying a full rebuild per write."

---

### Q54. Your index is 990MB and loaded per process. Fix it.

"Three options with different trade-offs.

Memory-map the index so multiple processes share physical pages — FAISS supports `IO_FLAG_MMAP`. Cheapest fix, keeps the current architecture, and cuts memory from N×990MB to roughly 990MB plus per-process overhead.

Or separate retrieval into its own service that API workers call over the network. Cleaner scaling because you scale retrieval and API independently, at the cost of a network hop.

Or move to a managed vector store and stop operating the index yourself.

For my scale, mmap is the right first move — it's a one-line change to how the index loads and solves the immediate problem. The service split is the right answer once retrieval and request handling need different scaling profiles."

---

### Q55. Design monitoring for this system. What pages someone at 3am?

**Tiers matter — say this explicitly.**

*Page immediately:* error rate above threshold, p99 latency breach sustained, index load failure, LLM provider circuit open. These are user-facing outages.

*Alert during business hours:* refusal rate shifted more than X% week-on-week segmented by reason (scope-guard versus no-evidence — my bug is why this is segmented), embedding-distribution drift, cache hit rate collapse, cost-per-query trending up.

*Dashboard only:* per-tenant QPS, retrieval score distributions, token consumption, sentiment class distribution over time.

**The one I'd add because of what I found:** refusal rate by reason, as a first-class metric. My scope guard falsely rejected a valid question and it was invisible in every accuracy metric because those questions never reach the model. A rise in out-of-scope rejections is the only signal that classifier is degrading.

**On alert fatigue:** anything that fires more than roughly once a week without action gets demoted to a dashboard. An alert nobody acts on trains people to ignore alerts.

---

### Q56. How would you A/B test the metadata-tagging change in production?

"Randomise at the session level, not per query — per-query assignment means one analyst sees inconsistent citation formatting within a session, which confounds any UX read.

Primary metric can't be RAGAS, because my ablation shows the two metrics move in opposite directions and picking one pre-decides the result. I'd want a behavioural metric: rate at which an analyst clicks through to a source review, and — better — rate at which they change their stated conclusion after doing so. That measures whether citations do work.

Guardrail metrics: p99 latency, cost per query, and refusal rate.

Sample size driven by the effect I'd need to detect. My offline deltas are +0.118 and −0.202 at n=5, which is far too noisy to power a calculation from. I'd run the offline evaluation at n=200 first to get a stable effect-size estimate, then power the online test off that. Going straight to an online test from n=5 risks running an underpowered experiment for weeks and learning nothing."
