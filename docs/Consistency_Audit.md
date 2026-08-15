# Cross-Chapter Consistency Audit

Pre-edit audit of all six chapters against each other, the marking scheme, and the project code.
Nothing in this document has been changed in the chapters yet. Items marked **VERIFY** need your
decision or a number I cannot derive; items marked **FIX** are unambiguous and I can correct them
during the chapter edits.

---

## Part 1 — The rubric findings that matter most

The marking scheme changes the priority order. Three things stand out.

### 1.1 Ablation study is a named Distinction requirement — currently unmet **[VERIFY / ACTION]**

**Presentation of results is 15% of the overall mark.** The 70–79% (Distinction) band reads:

> "Has done comparison with appropriate baselines where required. **Ablation study is also done
> where appropriate.**"

The 80–100% band repeats it. The dissertation currently states in three places (Ch3 §3.6.6, Ch5
§5.6.2, Ch6 §6.3.1) that the ablation suite was **implemented but not executed** because of
free-tier rate limits.

This is the single highest-leverage item in the whole audit. The code already exists in
`scripts/evaluate_rag.py` (experiments 2, 3, 5, 7a). Two observations:

- **The cheapest genuine ablation is 7a, metadata tagging** — two conditions (tagged vs untagged)
  across five questions, so roughly 10 RAG calls plus scoring. That directly tests the mechanism
  Chapter 5 currently has to *argue* rather than measure ("the prompt-level grounding constraints
  and metadata tagging appear to work… but without the ablation results this cannot be attributed
  to those specific mechanisms with confidence"). Running it converts an argued claim into a
  measured one and removes a stated limitation.
- **Experiment 5, retriever comparison** (similarity vs MMR), is similarly small and would
  substantiate the MMR design choice justified on reasoning alone in Ch3 §3.6.4.

Even one executed ablation moves this criterion from "comparison with baselines" (60–69 band) into
the band that names ablation explicitly. If none can be run, the fallback is to reframe what
already exists: **the binary re-evaluation in §4.5.4 is an ablation in substance** — it removes the
neutral class and measures the contribution of that component to aggregate performance. It is
currently labelled a "diagnostic experiment". Naming it as an ablation, and saying so in the
chapter that presents results, costs nothing and is defensible.

### 1.2 No abstract **[FIX]**

Report style/Presentation is 10%. The 80–100% band requires "Abstract can be read independently."
There is currently no abstract. I will draft one after the chapter edits, per your brief.

### 1.3 References are incomplete in a way the rubric penalises directly **[VERIFY]**

Literature review is 20%. Both the 70–79% and 80–100% bands require references "complete and
correct", with the top band asking for "exemplary completeness". See Part 4 — roughly fifteen
Chapter 2 citations are not in `references.md`, and your own `references.md` says so explicitly in
its closing section.

---

## Part 2 — Numbering and cross-reference errors

All **FIX**, all in Chapter 4 unless stated.

| # | Location | Problem |
|---|---|---|
| 1 | §4.5.2 and §4.5.3 | **Table 4.7 is used twice** — "Destination of true-neutral reviews" and "McNemar pairwise comparisons". Later tables need renumbering through 4.12. |
| 2 | §4.5.1, §4.5.2 vs §4.5.4 | **Figures 4.4, 4.5 and 4.6 are each used twice.** §4.5.1 has Fig 4.4 = overall model performance; §4.5.4 has Fig 4.4 = VADER threshold sweep. Same collision for 4.5 and 4.6. |
| 3 | §4.5.3 vs §4.5.5 | **Figure 4.7 used twice** — McNemar significance, and category frequency. |
| 4 | Ch4 generally | **Figure 4.2 is never defined.** Ch4 jumps 4.1 → 4.3. |
| 5 | Ch5 §5.1.5 | Cites "the confusion matrices in **Figure 4.2**"; Ch4 numbers the confusion matrices **Figure 4.6**. Wrong even before the duplicate numbering is resolved. |
| 6 | §4.3.1 | "producing the distribution in **Section 4.2.2**" — the distribution is in §4.2.3. |
| 7 | §4.2.3 | "the diagnostic evaluation samples equally from each platform (**Section 4.5.3**)" — diagnostics are §4.5.4. |
| 8 | §4.6 | "it is what the evaluation in **Section 4.5.6** exercises" — §4.5.6 is retrieval; the RAG evaluation is §4.5.7. |
| 9 | Ch3 §3.6.5 vs Ch5 §5.3.3 | LLM-as-judge literature cited as "Section 2.2.3" in Ch3 and "Section 2.1.2" in Ch5. Ch2 introduces it in §2.1.2 and applies it in §2.2.3, so both are half-right; needs one convention. |

---

## Part 3 — Numerical and factual consistency

### 3.1 Verified consistent (no action)

Checked and correct — recording these so you know they were tested:

- Corpus totals: 449,858 + 178,883 + 13,951 = 642,692 ✓ (Tables 3.2 and 4.1 agree)
- Class balance: 218,799 + 212,503 + 211,390 = 642,692 ✓
- Topic counts: 69 + 18 + 34 = 121 ✓ (Ch1 §1.1, Table 4.4, Ch5 §5.2.1 all agree)
- Diagnostic sample: 600 per platform × 3 = 1,800 ✓
- Minimum topic size against the stated rule (Ch3 §3.6.2, ~0.15% bounded 30–300): Amazon
  57,931 → 86 ✓, Yelp 149,838 → 224 ✓, Twitter 8,890 → floor of 30 ✓. The rule reproduces the
  table exactly.
- RoBERTa platform spread "23 points": 0.7853 − 0.5544 = 0.2309 ✓
- DistilBERT platform spread "two points": 0.8505 − 0.8301 = 0.0204 ✓
- RoBERTa on Twitter beating LR by 14 and NB by 30 points ✓
- Naive Bayes trailing logistic regression by "4.5 points": 0.8033 − 0.7580 = 0.0453 ✓
- Ch5 §5.1.5 "roughly two thirds recovered by replacing Naive Bayes with a better linear model":
  (0.8033 − 0.7580) / (0.8244 − 0.7580) = 68% ✓ — a nicely precise piece of reasoning
- Retrieval speedup "5.9 times": 127.61 / 21.71 = 5.88 ✓
- **RAGAS context precision/recall**: Ch1 O6, Ch3 §3.6.5 and Ch4 Table 4.12 all promise and report
  four metrics. I checked this against the code because `scripts/evaluate_rag.py` explicitly
  *excludes* those two metrics — but the generation comparison was run with
  `evaluate/evaluate_llm_vs_rag.py`, which does compute both (lines 73–75, 429–430). **Consistent,
  no issue.** Worth knowing in case an examiner reads the repository.

### 3.2 The fine-tuned model was trained on a different corpus **[VERIFY — most serious finding in this audit]**

This section replaces an earlier concern about the "approximately a quarter" figure. I wrote
`evaluate/measure_split_overlap.py` to reproduce both splits and measure the overlap exactly. That
run produced a clean 25.0%, appearing to confirm your figure — but inspecting the saved outputs of
`notebooks/3.DistillBERTFineTuning.ipynb` shows the reproduction is not valid, and reveals a
larger issue.

**The notebook's own printed output records a corpus of 592,354 reviews. The dissertation reports
642,692.**

| | DistilBERT corpus (notebook output, cell 13) | Dissertation corpus (Ch3 §3.2.2, Ch4 §4.2.3) |
|---|---:|---:|
| Total | 592,354 | 642,692 |
| Negative | 218,274 (36.8%) | 218,799 (34.0%) |
| Neutral | **164,473 (27.8%)** | **212,503 (33.1%)** |
| Positive | 209,607 (35.4%) | 211,390 (32.9%) |

The difference is 50,338 rows, of which **48,030 are neutral**. The corpus was evidently
regenerated with additional neutral reviews *after* DistilBERT was fine-tuned — which is consistent
with Ch3 §3.5.1's "class balancing is applied during sampling", but means that balancing was not in
effect for the fine-tuned model.

Three consequences follow.

**First, the 88,854 figure is internally correct but belongs to a different corpus.**
592,354 × 0.15 = 88,853, matching the notebook's printed test partition of 88,854. Reproducing the
same split on the *current* parquet gives 96,404. So Ch4 §4.3.3 and §4.5.1 are accurate about what
was run; they are just describing a corpus the dissertation never characterises.

**Second, the disclosure in Ch3 §3.5.3 understates the problem.** It currently describes the issue
as separately generated splits from one corpus. The actual situation is two different corpus
versions with materially different class priors — the fine-tuned model saw 27.8% neutral where the
other four models were tested on 33.1% neutral. Since neutral is the hardest class for every model
in the study, this is not a neutral difference.

**Third, it touches a claim in Ch4 §4.2.3**, which states that near-balanced classes are "what makes
macro F1 a fair headline metric". That holds for the four models on the shared test set. It does
not hold for the fine-tuned model, whose corpus was not balanced in the same way.

**What I cannot determine:** the true contamination percentage, because the 592,354-row corpus no
longer exists on disk — `data/processed/reviews_unified.parquet` has since been regenerated. The
25.0% my script produced is an artefact of computing both splits on the current file. Unless you
have the earlier parquet or can regenerate it deterministically from `scripts/preprocess.py`, the
overlap is not recoverable.

**Options, in the order I would consider them:**

1. **Disclose the corpus difference and stop quantifying.** Rewrite Ch3 §3.5.3 to state plainly that
   the fine-tuned model was developed against an earlier corpus version of 592,354 reviews with a
   different class distribution, that the overlap with the shared test set cannot now be measured,
   and that its clean figure comes from its own held-out partition of that corpus. This is honest,
   costs nothing, and the rubric's Additional Notes explicitly reward acknowledgment of limitations
   (item 9). It does mean removing "approximately a quarter" from Ch5 and Ch6.
2. **Re-run the evaluation of DistilBERT on the current shared test set** and report both figures —
   the clean one from its own partition, and a contaminated one on the shared set clearly labelled
   as such. This makes the five-model table genuinely comparable for the first time, but the
   contamination remains.
3. **Regenerate the corpus and re-fine-tune.** Correct, and almost certainly not feasible in your
   remaining time.

I have changed nothing. This needs your decision before I edit Chapters 3, 4, 5 or 6.

`evaluate/measure_split_overlap.py` is left in the repository — it is still the right script to run
if you recover the original corpus, and it documents the reproduction attempt.

### 3.3 "Differences between models cannot be attributed to differences in data" **[FIX]**

This absolute claim appears in **four places** and is contradicted by the documented split issue:

- Ch1 §1.2, Objective 2 — "A shared dataset and split is what makes the comparison controlled,
  since differences between models cannot then be attributed to differences in data."
- Ch4 §4.1.3, success criteria — "a controlled comparison in which differences between models
  cannot be attributed to differences in data"
- Ch6 §6.1 — "using a single evaluation protocol so that differences between methods could not be
  attributed to differences in data"
- Ch1 §1.3 — "so that differences between models cannot come from differences in test data"

This is exactly the formulation your brief asked to replace. The defensible version keeps the
design intent while acknowledging the exception — the claim holds for the four models sharing the
split, and the fine-tuned model's clean figure comes from its own partition. I will rewrite all
four consistently.

### 3.4 VADER macro F1 appears as two different numbers in consecutive sentences **[FIX]**

Chapter 5 §5.1.3 opens: "VADER's macro F1 of **0.4381**…" then two sentences later: "Widening the
band raises macro F1 from **0.5094** to a maximum of 0.5353."

Both are correct — 0.4381 is the full test set (Table 4.5), 0.5094 is the balanced diagnostic
sample (Table 4.9). Chapter 4 §4.5.4 flags the switch explicitly; Chapter 5 does not, and reads as
a contradiction. Needs one clarifying clause.

The same applies to VADER neutral F1: **0.086** in Table 4.6 (full test set) against **0.253** in
§4.5.4 (balanced sample).

### 3.5 Taxonomy coherence figures read as contradictory **[FIX — wording only]**

Table 4.10 reports mean within-category coherence of **0.6334**. The three named categories
immediately after are all *below* it: 0.5678, 0.3997, 0.3767.

This is arithmetically possible — the text does say single-review categories score near 1.0 and
pull the mean up — but as written a reader hits three numbers below the average and assumes an
error. One sentence making the mechanism explicit fixes it. Chapter 5 §5.2.1 repeats the 0.6334
without the qualification and should carry it too, because an unqualified 0.6334 slightly
overstates the taxonomy's quality.

---

## Part 4 — References **[VERIFY]**

`references.md` contains 14 verified entries and ends with a section headed "Still to source for
Chapter 2" listing what is missing. That is an accurate self-assessment. Against the 20% literature
criterion, this is the largest single risk in the dissertation.

**Cited in Chapter 2, absent from `references.md`:**

Turan et al. (2024) · Mishra (2026) · Zhang et al. (2023) · Hoyle et al. (2021) · Hoyle et al.
(2022) · Doogan-Poet Sirakoff & Buntine (2024) · Gu et al. (2024) · Li et al. (2024) · Yu et al.
(2024) · Chen et al. (2024)

**Citations that are not citations:**

| Form used | Where | Problem |
|---|---|---|
| "(*Electronics*, 2024)" | §2.1.2 | Journal title only, no author |
| "(*Big Data and Cognitive Computing*, 2024)" | §2.1.2 | Journal title only, no author |
| "arXiv:2508.17994, 2025" | §2.1.2 | Bare identifier |
| "arXiv:2211.16285, 2022" | §2.1.2 | Bare identifier |
| "arXiv:2412.14486, 2024" | §2.1.2 | Bare identifier |
| "arXiv:2305.14587, 2023" | §2.1.3 | Bare identifier |
| "arXiv:2407.13193, 2024" | §2.1.2 | Bare identifier |
| "a study of Indonesian customer complaints" | §2.1.2 | Unattributed, but a specific figure (0.68 vs 0.51) is quoted from it |
| "work evaluating traditional, pre-transformer and transformer paradigms on the Amazon Magazine Subscriptions 2023 dataset" | §2.1.2 | Unattributed |
| "adversarial approaches that use gradient reversal" | §2.1.2 | Unattributed |
| "Financial sentiment study (2023)" | Ch5 Table 5.1 | Unattributed, quoted figures (87%, 0.85) |
| "Indonesian complaint study (2024)" | Ch5 Table 5.1 | Unattributed, quoted figures |

**Also check:** "Mishra (2026)" is dated in the future relative to submission. If it is a
forthcoming/online-first paper that is fine, but it needs the correct APA form for that status.

I will not invent any of these. Where I edit those passages I will mark them
**[REFERENCE TO VERIFY]** in place.

**Novelty claims needing support or softening** (your brief §11, and the rubric's note that novelty
is not required at MSc):

- Ch1 §1.1: "Studies that test the same models across several types of feedback at once are
  **uncommon**"
- Ch2 §2.1.4: comparisons against a standalone baseline are "**uncommon**"
- Ch5 §5.4.4 and Ch6 §6.2.2: "comparisons of the same off-the-shelf methods across three registers
  … are **uncommon**"

These are load-bearing — they are the stated gap. Chapter 2's critical review does support them
indirectly (it shows the reviewed studies are single-domain), so the fix is to tie each claim
explicitly to that evidence: "within the studies reviewed in Section 2.1.3, comparisons of this
kind did not appear" is defensible; "uncommon in the literature" is a claim about all literature.

---

## Part 5 — Statistical wording **[FIX]**

### 5.1 "Barely above chance" is not supportable

Two places — Ch4 §4.5.1 ("VADER performs barely above chance on a three-class task") and Ch5 §5.1.3
("VADER's macro F1 of 0.4381 is barely above chance").

No random baseline was computed. For a balanced three-class problem, chance-level macro F1 is
approximately 0.33; VADER's 0.4381 sits about 10 points above it. "Barely above chance" is both
unsupported and unnecessarily harsh on your own baseline — and it slightly undercuts your argument
in §5.1.3 that VADER's role as a floor is informative. I will replace with a factual comparison
against the learned models.

### 5.2 Significance and effect size are drawn from different samples

Ch5 §5.1.4 and Ch6 §6.2.1 both pair a **full-test-set effect size** ("roughly three points of macro
F1") with a **balanced-sample p-value** (p = 1.7 × 10⁻²¹) in the same sentence, without noting the
switch. Chapter 4 §4.5.3 is scrupulous about this; Chapters 5 and 6 inherit the numbers without the
caveat. Needs a clause tying each p-value to the 1,800-review sample.

### 5.3 Already correct — preserving as-is

Your Naive Bayes / RoBERTa wording is already properly hedged in all three places it appears
(Ch4 §4.5.3, Ch5 §5.1.1, Ch6 §6.3.1) — "cannot be distinguished", "the available evidence does not
support ranking one above the other", "not distinguishable on the available evidence, not
established equivalence". This matches the formulation your brief asked for. **I will not touch
it.** I will only add the sample identification where §5.1.1 omits it.

---

## Part 6 — Repetition map

Findings repeated across chapters, with the function each instance should serve. None of these
should be deleted; several should be shortened or re-angled.

| Finding | Ch1 | Ch2 | Ch3 | Ch4 | Ch5 | Ch6 | Action |
|---|---|---|---|---|---|---|---|
| Pipeline is instrument, not contribution | §1 intro, §1.2 | — | §3.1.1 | — | — | — | Ch1 states it twice within three paragraphs; cut one |
| Single-domain literature weakness | §1.1 | §2.1.3, §2.1.4 | §3.8 | — | §5.4.3 | §6.1 | Ch3 §3.8 and Ch6 §6.1 can compress — Ch2 owns the argument |
| RoBERTa platform dependence | — | — | — | §4.5.4, §4.7 | §5.1.1, §5.1.2, §5.9 | §6.2.1 | Ch5 states it three times; §5.9 can point back rather than restate |
| Neutral-label mismatch | §1.3 | — | §3.2.3 | §4.2.4 | §5.1.1, §5.1.4 | §6.3.1 | Fine — each does different work |
| Fine-tuning → stability not peak | — | — | — | — | §5.1.4, §5.9 | §6.2.1 | Ch5 §5.9 and Ch6 overlap heavily; compress Ch6 |
| Five-question RAG limitation | — | — | §3.6.6 | §4.5.7 | §5.3.3, §5.6.2 | §6.1, §6.3.1 | **Six instances.** Ch5 §5.6.2 can cross-refer to §5.3.3 |
| Self-grading judge | §1.3 | §2.2.3 | §3.6.5 | §4.5.7 | §5.3.3, §5.6.2 | §6.3.1 | **Seven instances.** Ch1 and Ch4 can be brief pointers |
| Taxonomy defects | — | — | — | §4.5.5 | §5.2.3, §5.6.3 | §6.2.2, §6.3.1 | Ch5 §5.6.3 largely restates §5.2.3 |
| Relevance vs faithfulness | §1.4 | §2.2.3 | §3.6.5 | §4.5.7 | §5.3.2, §5.7 | §6.2.1 | Ch5 §5.3.2 is the strongest; others should defer to it |
| Method-selection rule (check label definition) | — | — | — | — | §5.8 | §6.2.3 | Ch6 repeats Ch5 near-verbatim; Ch6 should compress |

The heaviest concentration is Chapter 6 §6.1–6.2, which re-argues Chapter 5 rather than
synthesising it. Your brief asks for Ch6 to be more concise than Ch5; at present §6.2.1's four
findings restate §5.9 almost line for line.

---

## Part 7 — Reproducibility gaps (10% of the mark)

Documentation for reproducibility is assessed across Chapters 3, 4 and 5. Chapter 3 is already
strong — seeds, hyperparameters, split ratios, model checkpoint names and library choices are all
present. The gaps are these, and I will insert `[AUTHOR TO ADD: …]` markers rather than guess:

| Missing | Where it belongs |
|---|---|
| Exact Python version | Ch3 §3.4 — currently "Python 3" |
| Hardware / GPU used for fine-tuning | Ch3 §3.6.1 or Table 3.3 — "10,545 s" is reported with no indication of what hardware |
| Key package versions | Ch3 §3.4 — `requirements.txt` is referenced but no versions are stated in text |
| Repository URL | Ch3 §3.4 or Ch4 §4.6 |
| Script execution order | Ch3 §3.3 — the five stages are described but the commands to reproduce them are not listed |
| DistilBERT split generation script | Ch3 §3.5.3 — the 70/15/15 split is described but the script producing it is not named, and I could not find it in `scripts/` |
| Zero-shot model checkpoint | Ch3 §3.6.3 — the model is named only in `references.md` as `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, not in the chapter |
| Language-detection library | Ch3 §3.5.2 — "using language detection", tool unnamed |

One consolidation opportunity: seeds are stated in Ch1 §1.3, Ch3 §3.4, Ch3 §3.5.3, Ch3 Table 3.3
and Ch4 §4.3.3. Chapter 3 should own this; the others can point to it.

---

## Part 8 — Figures

You said figures come later, so this is a check on whether each proposed figure earns its place,
plus gaps.

**Proposed and justified — keep:**
Fig 2.1 (method taxonomy) · Fig 3.1 (workflow) · Fig 4.1 (review length) · Fig 4.3 (training
curves) · confusion matrices · per-platform macro F1 · three-class vs binary · VADER threshold
sweep · category frequency · platform distribution by category · sentiment distribution by category

**Gaps worth filling** (both are named in your brief and both support arguments currently carried
by prose alone):

1. **RAG architecture diagram** — Ch3 §3.6.4 and Ch4 §4.4.4 describe retrieval → metadata tagging →
   prompt → generation → source IDs entirely in text. This is the pipeline the whole of RQ2/RQ4
   rests on, and the "Problem analysis and design choices" criterion (20%) rewards design
   discussion. A diagram would carry it better than the current paragraph.
2. **Worked grounded RAG response** — an actual question, the retrieved reviews with their platform
   /rating/sentiment tags, the generated answer, and the returned review IDs. RQ4 is *about*
   traceability; showing one instance demonstrates it in a way no metric does. This is the single
   most persuasive figure you could add, and the data for it already exists.

**Possibly redundant:** Fig 4.9 (sentiment distribution by category) and the §4.5.5 prose both make
the positive-reviews-in-complaint-categories point. Keep the figure, shorten the prose.

**Caption check:** current placeholders describe *what was plotted* but several do not say *what
the reader should see*. Distinction-level captions state the finding.

---

## Part 9 — Things I deliberately did not change

- **The Naive Bayes / RoBERTa non-significance wording**, which is already correct in all three
  locations (Part 5.3).
- **All reported numerical results.** No metric, p-value, count or percentage has been altered.
- **The "approximately a quarter" contamination figure** — flagged in Part 3.2, awaiting your
  decision, because correcting it requires either a computation I cannot run or a decision to
  soften.
- **Any citation.** Nothing has been added to `references.md`.
- **Chapter 5's analytical structure**, which is the strongest part of the dissertation and maps
  well onto the 15% "Critical review of results" criterion.

---

## Suggested order of work

1. **Decide on Part 3.2** (the corpus-version finding). This blocks Chapters 3–6.
2. **Decide on Part 1.1** (run one ablation, or reframe the binary re-evaluation as one).
3. Chapter-by-chapter edits, in order, per your brief. Chapter 1 can proceed independently of
   both decisions above, since it only needs the Objective 2 wording fixed (Part 3.3).
4. Abstract last, once the chapters are settled.

---

## Appendix — measured output

From `python evaluate/measure_split_overlap.py`, run against the **current** parquet
(642,692 rows). Recorded for transparency; see Part 3.2 for why this does not measure the
actual contamination.

```
Shared 80/20 test set        : 128,539 reviews
DistilBERT train partition   : 450,141 reviews
DistilBERT validation        :  96,147 reviews
DistilBERT test partition    :  96,404 reviews

  Seen during training     :   26,516  (20.6%)
  Seen during validation   :    5,619  (4.4%)
  Seen in either           :   32,135  (25.0%)
  Genuinely held out       :   96,404  (75.0%)
```

The notebook's actual run reported 414,884 / 88,616 / 88,854 on a 592,354-row corpus.
