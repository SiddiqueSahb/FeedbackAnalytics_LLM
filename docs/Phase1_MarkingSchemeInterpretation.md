# Phase 1 — Marking Scheme Interpretation

Source: "Mapping Chapters to Marking Scheme.pdf" (uploaded). Seven weighted
criteria, each independently banded 0-39 / 40-49 / 50-59 (Pass) / 60-69
(Merit) / 70-79 (Distinction) / 80-100 (Distinction + "IS of publishable
quality"). The dissertation is marked criterion-by-criterion, not as a single
holistic read — a weakness in one criterion does not average out against
strength in another, it just caps that criterion's own band.

## Rubric map: what Distinction (70-79%) and Publishable (80-100%) actually require

| Criterion | Weight | Chapters | 70-79% (Distinction) requires | 80-100% (Publishable) adds |
|---|---:|---|---|---|
| Literature / technology review | 20% | 1, 2 | Excellent review including **some research papers or advanced material**; complete/correct references; excellent critical analysis; **some discussion of relevance to the project** | A **range** of research papers; insightful discussion of literature's direct relevance; exemplary referencing; exceptional/nuanced critical analysis |
| Problem analysis & design choices | 20% | 3, 4 | Extensive problem analysis; excellent discussion of design issues; choices **justified/validated against external sources** (lit review or user study); **problem tackled is technically challenging**; both analysis and design must independently be excellent | Same, plus **demonstrated innovation in approach**; outstanding, original justification |
| Reproducibility | 10% | 3, 4, 5 | Excellent documentation; thorough discussion of challenging implementation aspects | Same, plus shows originality, high professionalism, or deep understanding of implications |
| Presentation of results | 15% | 3, 4, 5 | Structured presentation with main parts highlighted; **clear links to the experiments carried out**; comparison with **appropriate baselines**; **ablation study done where appropriate**; shows understanding of the impact of the technologies used | Same, plus **extensive** comparison with baselines; original, in-depth discussion; clearly discusses broader implications |
| Critical review of results | 15% | 5 | Excellent discussion of model/system performance; excellent justification and explanation of results; excellent comparison with existing research; high level of synthesis; excellent linkage back to Aim/Objectives/Lit Review; thorough comparison **highlighting superiority or uniqueness** of the approach; excellent critical evaluation of objectives | Outstanding on all of the above; shows originality; clearly discusses broader implications |
| Critical review of aim/objectives + future work | 10% | 5, 6 | Excellent evaluation of what was achieved against objectives; excellent discussion of future work; clear, strong link between documented results and objectives; excellent insight into future work | Outstanding on all of the above; originality; broader implications discussed |
| Report style | 10% | all | Good design, logically structured, balanced presentation, clear and readable | Excellent design; **abstract can be read independently and gives a comprehensive summary** |

## Notes that materially change how strict to be

1. **A literature-review-type project is explicitly allowed**, but a "simple" literature review will not score highly — the rubric wants "comprehensive critical comparison and evaluation... including a well-justified recommendation/conclusion." This is echoed in the Ch1&2 row's repeated emphasis on *critical analysis*, not just coverage.

2. **Novelty is explicitly NOT required for MSc** — the PDF states this outright ("The requirement for novel research is for PhD only... Students shouldn't be penalised for not doing novel research"). This matters a lot for how Chapter 1's contribution claims and Chapter 6's synthesis should be framed: claims should be **proportionate**, not apologetic about "just" being a comparative/applied study, and should not be inflated into false novelty claims either. The nine listed "sample expectations for novelty" (unique problem/gap, creative application of existing methods, emerging tech, interdisciplinary integration, novel algorithms, novel context, ethical/societal implications, validation through results, honest limitations) are framed as *optional* enhancers, not a checklist to force-fit.

3. **"Ablation study done where appropriate" appears at both the Distinction and the row above it (Merit)** under Presentation of Results — this is the rubric's own hook for the RAG ablation-suite gap already flagged in the earlier `docs/RAG_Hallucination_Audit.md`/Consistency Audit work: if ablation code exists but wasn't executed, Chapter 4/5 must not claim it as delivered evidence. This is a direct, named risk to the 15%-weighted Presentation-of-Results criterion specifically.

4. **Problem analysis & design choices (20%, the single largest criterion alongside Lit Review) requires design choices to be validated against *external sources*** — i.e. every non-trivial design decision (model choice, split strategy, retrieval design, prompt design) needs an explicit tie back to literature or the marking scheme won't move past Merit regardless of how well the choice itself is explained.

5. **Report style's top band requires an independently-readable abstract.** Worth checking early rather than last, since it's a quick, contained fix if missing.

6. Distinction bands are explicitly the *minimum* required for "MAY BE of publishable quality" — the wording literally reads "The work MAY BE of publishable quality" at 70-79% and "The work IS of publishable quality" at 80-100%. So hitting 70-79% per criterion is the realistic, evidence-honest target for this audit; 80-100% is a stretch goal that depends on originality/insight quality, not just correctness.

## What this means for the audit that follows

The two criteria most exposed by what's already been found in this project (per `docs/Consistency_Audit.md` and `docs/RAG_Hallucination_Audit.md`) are:

- **Presentation of results (15%)** — because of the RAG ablation implemented-vs-executed gap, and because Chapter 4 has known duplicate table/figure numbering that directly damages "structured presentation."
- **Problem analysis & design choices (20%)** — because several design choices (e.g. FAISS distance-metric choice, MMR+threshold retrieval design, the DistilBERT split strategy) need to be explicitly and correctly tied to literature, and the DistilBERT corpus-version mismatch found earlier is exactly the kind of thing that, left unexamined, would read as a design flaw rather than a disclosed, understood limitation.

Both are fixable through accurate reporting and honest framing — neither requires new experiments, consistent with the brief's restriction against inventing results.
