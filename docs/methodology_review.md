# Complaint Categorisation Methodology Review
FeedbackAnalytics_LLM — MSc Dissertation

## Root cause, confirmed from your actual code

I read `scripts/discover_categories.py`, `nlp/categoriser.py`, and the three JSON outputs in `data/processed/`. There is a concrete, verifiable bug that explains both of your examples, on top of the deeper methodology issues.

`discover_categories.py` runs BERTopic **separately per platform** and writes `complaint_categories_amazon_negative.json`, `complaint_categories_twitter_airline_negative.json`, `complaint_categories_yelp_negative.json`. No `complaint_categories_all_negative.json` exists in your `data/processed/` folder.

`categoriser.py`'s `_load_categories()` checks for that consolidated file first (Priority 1), doesn't find it, and falls to Priority 2: it **globs all three per-platform files and concatenates them into one candidate-label pool**, capped at `MAX_CATEGORIES = 20`. That means every single review — Amazon, Yelp, or airline — is scored by BART against a mixed pool that includes `"Fees & Charge Problems"`, `"Store Operational Hours Failures"`, and `"Travel Experience Failures"` alongside `"Hardware Compatibility Issues"`.

This is exactly why "The battery stopped charging after one week" gets matched to `Store Operational Hours` and `Fees & Charge Problems` — those labels are literally sitting in the candidate list for an Amazon electronics review. There's a second contributing effect: BART-MNLI's entailment scoring is somewhat lexically anchored, so `"charging"` (battery) and `"charge"` (fee) share a surface token, giving `Fees & Charge Problems` a spurious boost it wouldn't get from a purely semantic model.

Fixing this file-loading bug alone would meaningfully improve your examples. But it's a symptom of a deeper design gap, addressed below.

---

## 1. Should BERTopic topics be directly converted into BART candidate labels?

No — this is methodologically weak, independent of the cross-platform bug.

A BERTopic topic is a statistical artifact: a cluster of documents grouped by embedding proximity, summarised by a c-TF-IDF word distribution. Its boundaries are set by hyperparameters (`min_topic_size`, UMAP `n_neighbors`, HDBSCAN `min_samples`, `nr_topics`) that control *clustering resolution*, not *business meaningfulness*. Two different corpus samples of the same domain will produce a different number of topics with different boundaries, even though the underlying complaint types haven't changed.

Turning each topic into one candidate label 1:1 inherits every clustering imperfection directly into your classification taxonomy:

- **Over-fragmentation**: your own `twitter_airline` output has `"Flight Boarding Delays"`, `"Flight Schedule Failures"`, `"Flight Schedule Delays"`, `"Flight Rescheduling Failures"`, `"Flight Rescheduling Issues"`, `"Travel Itinerary Delays"` — six labels for what a business analyst would likely call one category, "Flight Delay & Rescheduling Issues." Likewise `"Travel Experience Failures"` / `"Travel Experience Issues"` / `"Travel Disruption Issues"` / `"Trip Disruption Issues"`.
- **No mutual exclusivity**: zero-shot NLI classifiers estimate each label independently and pick the argmax. When several labels describe near-identical concepts, probability mass that should concentrate on "the" correct label gets split across its near-synonyms, systematically lowering every one of their scores relative to an unrelated-but-distinct label.
- **Granularity mismatch**: topic count is driven by corpus size and clustering hyperparameters, not by how many distinct complaint types actually exist. A bigger sample yields more topics for statistical reasons alone, not because the domain got more complex.

A topic should be treated as *evidence toward* a taxonomy, not *the* taxonomy.

---

## 2. One category per topic, vs. merge into 15–20 business categories, then BART

The merge approach is academically stronger — but only if the merge step is a defensible algorithm, not "ask an LLM to shorten the list."

Why merging is the correct approach:

- It directly targets the failure mode in Section 1: near-duplicate topics get collapsed before they ever reach the classifier, instead of being laundered into fluent-but-redundant English by an LLM that can't see them.
- Zero-shot NLI models are documented to degrade as candidate-label count grows and as label semantic distance shrinks (this is well established in zero-shot text classification literature, e.g. Yin et al. 2019). A smaller, well-separated label set is not a nicety — it's a precondition for the classifier to work as intended.
- It gives you a taxonomy that satisfies the two properties classification requires: mutual exclusivity and (approximate) exhaustiveness. One-per-topic can't provide either, because BERTopic's clustering has no notion of either property.
- It produces a much stronger dissertation narrative: "topics were merged via [named algorithm] with [quantified threshold], validated via [coherence metric / manual audit]" is defensible methodology. "An LLM turned each topic into a label" is a single unvalidated transformation step with no way to catch its own errors.

The merge step itself should not be delegated to an LLM's judgement about how many categories to produce. It should be a clustering step over topic-level embeddings (BERTopic exposes `topic_model.topic_embeddings_` natively — you don't need to hand-roll centroid computation), with the target category count chosen and justified (via a dendrogram cut point, a coherence metric, or a small manual validation pass), and the LLM used only afterward, to *name* the already-merged clusters.

---

## 3. Is BART the correct model? Alternatives

BART-large-MNLI zero-shot (entailment: "this text is about {label}") is a reasonable, standard **baseline** and fine to keep as one arm of a comparison. As your sole production classifier it has three concrete weaknesses relevant to your failures:

1. **Lexical anchoring**: MNLI-trained models pick up on shared surface tokens between premise and hypothesis even when the underlying sense differs — the "charging" vs. "charge" collision is a textbook case.
2. **Degradation with label count**: BART-MNLI does one binary entailment pass per candidate label and normalises; with 20 candidates, especially semantically close ones, small embedding noise flips the argmax.
3. **Bare labels are weak hypotheses**: `"Fees & Charge Problems"` as a 3–4 word label carries far less discriminating signal as an NLI hypothesis than a full sentence description would.

Alternatives, and where each fits:

- **Sentence-Transformers embedding similarity (bi-encoder)**: embed the review and each category (ideally a description + a few exemplar phrases, not just the label), cosine-rank. Fast, scales to hundreds of categories, less prone to lexical-token bias than NLI hypothesis templates because contrastive-trained embedding models weight paraphrase/semantic similarity more than raw token overlap. Weaker at fine-grained distinctions than a cross-encoder.
- **Cross-encoder / NLI reranker on a shortlist ("retrieve then rerank")**: use the bi-encoder to cut 20+ categories down to a top-5 shortlist, then run NLI (BART or a stronger DeBERTa-v3 zero-shot model) only on that shortlist. This is the standard way to make zero-shot NLI robust to large label sets, and it reuses the `all-MiniLM-L6-v2` model you already have loaded for BERTopic — no new dependency.
- **DeBERTa-v3 zero-shot** (e.g. `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`): same `pipeline("zero-shot-classification", ...)` API as BART, near drop-in, and consistently benchmarks above BART-large-mnli on zero-shot classification tasks, particularly with larger label sets. Worth using as your primary model, with BART kept as the reported baseline for comparison.
- **SetFit (few-shot Sentence-Transformer fine-tuning)**: if you're willing to hand-label ~8–16 examples per category, SetFit substantially outperforms zero-shot with very little labelling effort, while still fitting a "light supervision, not full supervision" methodology narrative.
- **Direct LLM classification** (structured prompt with the review + shortlist + few-shot examples): strong in practice, but weaker as a dissertation contribution on its own — more valuable as one arm of a comparison than as the sole method, since it's harder to analyse/explain than an NLI or embedding approach.

Recommended primary architecture: **embedding retrieval (shortlist) → NLI rerank on the shortlist (DeBERTa-v3-zeroshot, BART reported as baseline)**. This resolves the "too many candidate labels" problem structurally instead of by an arbitrary hard cap, and gives you a natural ablation table for your evaluation chapter.

---

## 4. Methodological weaknesses in `discover_categories.py`

In order of impact:

1. **Cross-platform taxonomy silos with no reconciliation step.** Categories are discovered per-platform and never merged into one taxonomy before consumption — this is what produces the contamination bug in `categoriser.py`. Root cause, not a downstream symptom.
2. **Deduplication is exact-string-match only** (lines ~504–513). `"Flight Rescheduling Failures"` and `"Flight Rescheduling Issues"` are treated as two different categories. This is the direct cause of your 40–60 category counts.
3. **LLM naming is called per-topic in isolation** (`llm_convert_to_category`, called once per topic in the loop at line ~376). Each call has zero visibility into labels already generated, so it cannot avoid producing near-synonyms as topic count grows — semantic drift is guaranteed, not incidental.
4. **The prompt fixes surface grammar, not concept boundaries.** It mandates labels end in "Issues/Problems/Failures/Delays" but gives the LLM no constraint on how many distinct *concepts* should exist — so it happily produces four grammatically-varied labels for one concept, satisfying the prompt while defeating its purpose.
5. **No target category count.** `nr_topics="auto"` plus `dynamic_min_topic_size()` (0.5% of corpus, capped 50–500) determines category count purely from corpus size and HDBSCAN's internal reduction heuristic — unrelated to how many genuinely distinct complaint types exist. Category count becomes an emergent side effect of sample size, not a deliberate methodological choice, and is inconsistent across platforms for reasons unrelated to complaint diversity.
6. **`filter_noise_topics` thresholds (0.5 / 0.6 / 0.7) are hand-picked and unvalidated.** Reasonable heuristics, but with no coherence metric or ablation reported, they're presented as methodology while actually being untuned constants.
7. **No topic-quality metric anywhere in the script.** No coherence score (e.g. NPMI/c_v), no diversity metric, no report of inter-topic similarity. You cannot defend `min_topic_size` or the noise-filter thresholds quantitatively without one.
8. **The two labelling paths are different methodologies bundled in one script, and the non-default one is what you're actually using.** `--use-llm` is off by default (`clean_topic_label` rule-based path runs unless the flag is passed), but your reported example categories are clearly LLM-generated. Make sure your dissertation's stated methodology matches the flag combination actually run.
9. **No platform/domain metadata carried on categories.** A category has no marker saying "this only makes sense for airline reviews" — so nothing downstream can prevent an airline-only category from being offered as a candidate for an Amazon review, beyond the accident of which files happen to exist on disk.
10. **No linkage back to representative example reviews per category**, only keyword lists — makes qualitative validation of label faithfulness (which you'll want for a dissertation appendix) a manual, separate step rather than a built-in output.
11. **No category-frequency signal reaches the classifier.** A category discovered from a 289-review minimum cluster and one representing thousands of reviews are treated identically as "one candidate label out of 20" at inference time.

---

## 5. Proposed architecture

**Stage A — Sentiment.** Unchanged: fine-tuned DistilBERT.

**Stage B — Topic discovery.** BERTopic per platform as now (keep this — platform-specific runs surface platform-specific vocabulary better than one giant mixed run), but every topic retains: keywords, `topic_model.topic_embeddings_` centroid, member document count, source platform, and 2–3 example reviews.

**Stage C — Taxonomy construction (new stage, currently missing entirely).**
1. Pool topic centroids from all platform runs.
2. Hierarchical/agglomerative clustering on centroid cosine distance to merge near-duplicate topics into a target ~15–25 canonical clusters (BERTopic's own `reduce_topics()` / `hierarchical_topics()` utilities do this natively — no need to hand-roll). Choose the cut point via dendrogram inspection or a coherence metric, and report it.
3. One LLM call per **merged cluster**, with the full running list of category names already assigned included in the prompt (or one batched call covering all clusters at once) so duplicates are structurally prevented rather than hoped-away. Ask for a category **name + one-sentence description + 3–5 exemplar phrases**, not just a name.
4. Manual spot-check: sample a handful of reviews per merged category and confirm label faithfulness — a small, reportable validation step.
5. Persist **one** consolidated `complaint_categories.json` (name, description, exemplars, source platforms, member count) — not per-platform files merged naively downstream.

**Stage D — Classification (reworked `categoriser.py`).**
1. At startup, embed each category's description+exemplars with the same `all-MiniLM-L6-v2` model already used for BERTopic; cache.
2. Per review: embed the text, cosine-rank against all categories, take top-K (~5) shortlist.
3. Run zero-shot NLI (DeBERTa-v3-zeroshot as primary, BART reported as baseline) only on the shortlist, using the category **description** as the hypothesis, not the bare label.
4. Confidence threshold, calibrated against a small labelled validation set: below threshold → "Unclassified / Emerging Complaint" rather than a forced wrong label. Doubles as lightweight novelty detection.

**Stage E — FastAPI.** Same contract (sentiment + categories), extended with description/confidence/matched-exemplar in the response for explainability — a natural addition for a dissertation's interpretability discussion, and low cost given Stage D already computes it.

**Missing piece that matters most for the dissertation itself:** nothing in the current repo quantitatively evaluates complaint-categorisation quality (`data/results/` only holds sentiment-model evaluation artifacts). Build a small gold set (150–300 manually labelled reviews across all platforms) before making further changes — it's the only way to claim, with evidence, that any of the above actually improved accuracy.

---

## 6. Exact changes to `discover_categories.py` (algorithm, not code)

- Run per-platform BERTopic as now, but stop treating each run's output as final — collect all `topic_details` across platforms before any labelling happens.
- Use `topic_model.topic_embeddings_` (already computed by BERTopic) as each topic's centroid, instead of relying on keyword-bag similarity alone.
- Merge topics via agglomerative clustering on centroid cosine distance (or BERTopic's `reduce_topics()`/`hierarchical_topics()`), with the merge threshold or target count chosen from a dendrogram cut or coherence metric, replacing the current `nr_topics="auto"` black box.
- Replace the per-topic-isolated `llm_convert_to_category` loop with a single call (or a sequential loop that passes forward the running list of already-assigned names) operating on the **merged** clusters, requesting name + description + exemplar phrases per cluster, so the model can see and avoid duplicates.
- Add an explicit target-category-count parameter driven by the merge step (e.g. `--target-categories 18`), replacing the implicit, corpus-size-driven count.
- Write one consolidated `complaint_categories.json` as the production artifact; keep the per-platform raw topic dumps as intermediate/audit files only (useful for a dissertation appendix, not as classifier input).
- Report a coherence/separation metric (e.g. mean pairwise cosine similarity between final category centroids) alongside the taxonomy, so `min_topic_size` and merge-threshold choices are empirically justified rather than asserted.

---

## 7. Exact changes to `categoriser.py`

- Remove the Priority-2 "glob every per-platform file and concatenate, cap at 20" fallback entirely. Load the single consolidated taxonomy only. This alone fixes the contamination behind both of your examples.
- Load category description + exemplars, not bare label strings; precompute and cache their embeddings at startup.
- Add the retrieval shortlist step (embed review, cosine-rank, top-K) before the NLI call — this bounds classifier input size structurally as the taxonomy grows, instead of silently truncating categories 21+ via `MAX_CATEGORIES`.
- Swap to (or add as a comparison arm) DeBERTa-v3-zeroshot; keep BART as the reported baseline.
- Use the category description as the NLI hypothesis text instead of the bare label.
- Add a confidence threshold and "Unclassified" fallback, calibrated on a held-out labelled sample.
- Add an evaluation harness that scores the pipeline against the gold set (accuracy / macro-F1), so every subsequent change to either script can be measured rather than eyeballed.

---

## 8. Is the methodology fundamentally wrong?

The core idea — data-driven complaint discovery via BERTopic feeding a zero-shot classifier, instead of hardcoded categories — is sound and is legitimate MSc-level methodology. Keep it.

What's actually wrong is narrower but structural: BERTopic's statistical output is being treated as a finished classification taxonomy with no taxonomy-construction step in between, and platform-siloed category files are being concatenated at inference time with no reconciliation. Neither is fixable by tuning BART's parameters or capping candidate count — both require inserting a genuine methodological stage (semantic merge + validated taxonomy) that doesn't currently exist, and consolidating categories into one universal set before they reach the classifier.

Keep: DistilBERT sentiment, BERTopic discovery, LLM-assisted naming, zero-shot classification, FastAPI serving.
Add: taxonomy merge/consolidation stage, richer category representations (description + exemplars), retrieval-based shortlisting before NLI, a gold evaluation set, one universal taxonomy.
Reconsider: BART as sole classifier — either swap to DeBERTa-v3-zeroshot or, better for the dissertation, report a comparative ablation across methods. That turns a bug fix into an evaluation chapter.

---

## 9. Implementation roadmap

**Phase 0 — Build the yardstick.** Manually label 150–300 negative reviews across all three platforms with the correct complaint category. Nothing after this phase can be evaluated without it.

**Phase 1 — Quick fix, measure baseline.** Make `categoriser.py` load one consolidated file only (even a manually merged `complaint_categories_all_negative.json` as a stopgap). Re-run your two failing examples, confirm improvement, record accuracy on the Phase 0 gold set as your baseline.

**Phase 2 — Topic merge.** Implement centroid-based hierarchical merging using `topic_model.topic_embeddings_`; target ~15–20 canonical categories; validate via dendrogram + manual spot-check.

**Phase 3 — Taxonomy-aware LLM naming.** Rework the naming step to operate on merged clusters with shared context (no duplicate names), producing name + description + exemplars.

**Phase 4 — Rework the classifier.** Add embedding-based shortlist retrieval, description-based NLI hypotheses, confidence threshold + unclassified bucket.

**Phase 5 — Model ablation.** Compare BART-flat (current), BART+taxonomy-fix, DeBERTa+taxonomy-fix, and SBERT-retrieval+DeBERTa-rerank (and SetFit, if time allows) against the gold set. Report precision/recall/macro-F1 per method — this is your results/evaluation chapter.

**Phase 6 — Wire into FastAPI.** Final chosen pipeline, response extended with description/confidence/matched-exemplar.

**Phase 7 — Write-up.** Methodology chapter: discovery → merge → taxonomy → classification, with every hyperparameter justified by a Phase 2/5 metric. Results chapter: the ablation table. Discussion: interpret concrete error cases (e.g. the "charging" vs. "charge" lexical collision) as a qualitative finding about zero-shot NLI limitations — a documented limitation reads better to examiners than a silent bug.

---

*Note, unrelated to the above: `config.py` currently has live-looking Groq and Google API keys hardcoded as default values rather than pulled only from `.env`. Worth moving those out before this repo goes anywhere public (e.g. GitHub), independent of the modelling work.*
