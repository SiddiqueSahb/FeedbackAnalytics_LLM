# Chapter 2 — Literature Review

This chapter reviews the research that the present study builds on and argues with. Its purpose is not to catalogue what has been published, but to work out what is already settled, what is still contested, and where the evidence needed to choose between methods is missing. That last point is what the dissertation responds to, so the review is written to lead into it.

The chapter is organised in three parts. Section 2.1 reviews the substantive literature: why customer feedback analysis matters, how the problem has been tackled, how the competing studies compare, and what gap remains. Section 2.2 examines the supporting literature behind the methodological decisions taken here, covering datasets and evaluation design. Section 2.3 reviews the tools and technologies themselves and justifies the specific choices made. Section 2.4 summarises and connects the review to the methodology in Chapter 3.

Emphasis falls on work published since 2019, since transformer-based methods and retrieval-augmented generation have both changed substantially within that window. Older work is cited only where it introduced something still in active use.

---

## 2.1 Literature Review

### 2.1.1 Research context

Customer feedback is unusual among data sources in being simultaneously abundant, unsolicited and directly actionable. A dissatisfied customer writing a review is describing a specific failure without being prompted, which is information that surveys struggle to obtain. The difficulty is volume. Platforms such as Amazon and Yelp hold review counts in the tens of millions, and no analyst team reads more than a sample.

Academic interest has followed the commercial interest closely. Sentiment analysis grew into one of the largest subfields of applied natural language processing, and the surveys of the period document a steady migration from lexicon and feature-based methods towards learned representations (Pang & Lee, 2008; later work summarised in Section 2.1.2). More recently the arrival of large language models has pushed the field towards generation: instead of classifying feedback, systems are now expected to summarise it, answer questions about it, and recommend action.

The relevance to industry is immediate, and the reason is specific. Sentiment classification supports monitoring; it tells an organisation how bad things are. Topic and complaint discovery supports diagnosis; it tells them what is wrong. Question answering over the corpus supports investigation; it lets an analyst pursue a particular concern. These are different jobs with different failure modes, and treating them as one problem is part of why method selection in this area is often poorly evidenced.

Figure 2.1 shows how the literature reviewed in this chapter maps onto those three jobs, and where each method family sits.

*(Figure 2.1 — Taxonomy of approaches to customer feedback analysis. A three-column diagram: column one, Sentiment classification, listing lexicon-based, classical ML, pretrained transformer and fine-tuned transformer; column two, Complaint discovery, listing LDA, NMF and BERTopic; column three, Feedback exploration, listing standalone LLM and retrieval-augmented generation. Evaluation approaches appear beneath each column: supervised metrics, coherence measures, and reference-free generation metrics respectively. To be produced and inserted here.)*

### 2.1.2 Current state of the art

**Sentiment classification.** The dominant approach is now the fine-tuned transformer. BERT established the pattern of pretraining a bidirectional encoder and adapting it to a downstream task (Devlin et al., 2019), and the variants that followed traded accuracy against cost in different ways. RoBERTa improved pretraining without changing the architecture (Liu et al., 2019), while DistilBERT reduced parameters by roughly 40% through distillation while retaining most of the performance (Sanh et al., 2019). The practical consequence is that a strong classifier is now a fine-tuning job rather than a research project.

Recent empirical work on review data supports the general ranking. A 2024 comparative study of Amazon product sentiment across machine learning, deep learning and transformer-based techniques reported transformer models outperforming both earlier families **[REFERENCE TO VERIFY — currently cited as journal title only, "(*Electronics*, 2024)"; needs author, year, title, volume, DOI]**. A parallel study comparing machine learning against pre-trained language models on product reviews reached a similar conclusion **[REFERENCE TO VERIFY — currently "(*Big Data and Cognitive Computing*, 2024)", journal title only]**, as did work evaluating traditional, pre-transformer and transformer paradigms on the Amazon Magazine Subscriptions 2023 dataset using several embedding schemes **[REFERENCE TO VERIFY — no citation given]**.

Two things about this body of work matter for the present study. First, the comparisons are consistently run within a single domain, most often Amazon product reviews. Second, they report aggregate accuracy or weighted F1 far more often than per-class behaviour. Both choices make transformer superiority easier to demonstrate and harder to qualify.

**Domain adaptation and transfer.** The assumption that a pretrained transformer transfers well is itself an empirical claim, and the models are not domain-neutral. The `twitter-roberta` family, for instance, was pretrained on tweets and benchmarked on Twitter-specific tasks (Barbieri et al., 2020), which is the reason it performs strongly on short social text. Whether that advantage survives a move to long-form review prose is exactly the kind of question single-domain studies cannot answer.

A separate literature does address this directly, though from a different angle. Cross-domain sentiment classification treats the mismatch as a transfer-learning problem, seeking representations that hold across source and target domains. Recent work in this line includes feature projection with multi-source attention, designed to reduce negative transfer when knowledge is drawn from several source domains at once (Zhang et al., 2023) **[REFERENCE TO VERIFY — cited in text, absent from the reference list]**, and adversarial approaches that use gradient reversal to encourage domain-invariant representations across Amazon, Yelp and IMDB reviews **[REFERENCE TO VERIFY — no citation given]**. The field's own framing is instructive: it treats domain shift as a serious obstacle requiring dedicated architecture, which sits awkwardly beside the assumption in applied comparison papers that an off-the-shelf pretrained model will simply work on whatever data it is given. The present study takes the applied position but tests it, measuring performance per platform instead of assuming transfer holds.

**Applied feedback analysis with large language models.** A newer strand applies language models directly to commercial feedback. Recent work has proposed multi-model pipelines for automating e-commerce feedback analysis, and weighted ensembles combining GPT-family models with deep learning for multilingual review classification. Aspect-based sentiment analysis has attracted particular attention, since it promises the granularity that document-level sentiment lacks. Findings here are mixed in a way that matters: comparisons of large language models against smaller fine-tuned models such as BERT report that the larger models handle simple sentiment classification competently but struggle with tasks requiring structured or nuanced output, while outperforming in few-shot settings where annotation is scarce. Purpose-built resources are beginning to appear, including a retail corpus for aspect-based sentiment analysis with large language models **[REFERENCE TO VERIFY — bare identifier arXiv:2508.17994; needs author, year, title]**. The pattern suggests that model scale does not straightforwardly substitute for task-specific training, which is consistent with the position taken in this dissertation.

**Zero-shot classification.** Assigning text to categories that were not seen during training is usually framed as natural language inference: each candidate label is verbalised as a hypothesis and scored against the text as premise (Laurer et al., 2023). The approach matters here because complaint categories discovered by topic modelling have no labelled examples by construction. Evaluations of unsupervised text classification have compared entailment-based and similarity-based approaches directly **[REFERENCE TO VERIFY — bare identifier arXiv:2211.16285]**, and more recent benchmarking has extended the comparison across cross-encoders, embedding models, rerankers and generative models **[REFERENCE TO VERIFY — no citation given]**. The consistent finding is that the quality of the label verbalisation matters as much as the model, which is why the pipeline in this study uses a full category description as the hypothesis instead of a bare category name.

**Complaint and topic discovery.** LDA remained the default for two decades (Blei et al., 2003), but it depends on word co-occurrence within documents and degrades on short informal text. BERTopic replaced the probabilistic machinery with an embedding-and-clustering pipeline, using a sentence transformer for representation, UMAP for dimensionality reduction, HDBSCAN for clustering, and class-based TF-IDF for topic description (Grootendorst, 2022).

Comparative evidence since then has been broadly favourable but more nuanced than the headline suggests. Turan et al. (2024) compared LDA, NMF and BERTopic on an Amazon product review dataset and concluded that all three were useful, with BERTopic producing more interpretable topics **[REFERENCE TO VERIFY — cited throughout Chapters 2 and 5, absent from the reference list]**. Mishra (2026) reached a similar conclusion in marketing research **[REFERENCE TO VERIFY — absent from the reference list, and dated after submission; if forthcoming or online-first, use the APA form for that status]**. Work comparing unsupervised topic modelling techniques for qualitative analysis of online communities found BERTopic produced more compact and better separated clusters, though not uniformly better coherence scores **[REFERENCE TO VERIFY — bare identifier arXiv:2412.14486]**. Notably, a study of Indonesian customer complaints reported LDA coherence of 0.68 against BERTopic's 0.51, with a hybrid model outperforming both — a reminder that BERTopic does not dominate on every dataset or metric **[REFERENCE TO VERIFY — unattributed, and these two figures are benchmarked against in Chapter 5, Table 5.1; this citation is load-bearing]**.

**Retrieval-augmented generation.** RAG was introduced to combine parametric knowledge in model weights with non-parametric knowledge retrieved at inference (Lewis et al., 2020). Its adoption accelerated sharply once hallucination was recognised as a systematic property of generative models, not an occasional defect (Ji et al., 2023). Recent surveys document the resulting proliferation of architectures and evaluation schemes (Yu et al., 2024) **[REFERENCE TO VERIFY — Yu et al. cited in Chapters 2 and 5, absent from the reference list; a second survey is cited only as the bare identifier arXiv:2407.13193]**.

Evaluation has become the contested part. RAGAS proposed reference-free metrics for faithfulness, answer relevance and context relevance, allowing evaluation without human annotation (Es et al., 2024). Benchmarking work has shown that large language models remain sensitive to noise and irrelevant passages in retrieved context (Chen et al., 2024) **[REFERENCE TO VERIFY — absent from the reference list]**. Yu et al. (2024) note that many benchmarks collapse performance into a single aggregate score, which is inadequate for diagnosing whether a failure originated in retrieval or generation.

**Evaluation by language model.** Because reference-free metrics are usually computed by a language model acting as judge, the reliability of that judge is now itself a research area. Surveys published since late 2024 catalogue position bias, verbosity bias and self-enhancement bias, where a model rates its own outputs more favourably (Gu et al., 2024; Li et al., 2024) **[REFERENCE TO VERIFY — both cited in Chapters 2, 3 and 5, neither in the reference list; these support the study's own stated judge-bias limitation, so they are load-bearing]**. This literature bears directly on a limitation of the present study, discussed in Section 2.2.3.

### 2.1.3 Critical review

Comparing these studies exposes patterns that individual papers do not.

**Datasets.** The overwhelming majority of sentiment comparisons use a single source, and Amazon reviews dominate. This is convenient — the data is large, labelled by star rating and freely available — but it produces a narrow evidence base. Review length, formality and complaint style differ substantially between a product review, a restaurant review and a complaint tweet. A study establishing that a transformer wins on Amazon has not established that it wins on customer feedback.

There is also a labelling convention worth questioning. Deriving sentiment labels from star ratings is standard practice, and it is what makes datasets of this scale possible. But it is an approximation, since a four-star review can contain a specific complaint, and papers rarely discuss the ceiling this places on achievable accuracy.

**Algorithms.** The comparisons are usually generous to the newest method. Transformer baselines are typically fine-tuned; classical baselines are often not tuned to a comparable degree. Where classical models are given careful feature engineering and regularisation they perform closer to transformers than the headline numbers imply, particularly on longer, well-formed text where lexical features are informative.

**Evaluation metrics.** This is the weakest link in the literature. Accuracy and weighted F1 dominate, and both conceal class-level failure. A three-class problem in which a model performs well on two classes and collapses on the third can still report a respectable aggregate. Macro F1 partially addresses this, and confusion matrices address it fully, but neither is reported consistently.

For topic modelling the situation is considerably worse, and the problem is not merely that studies disagree about which coherence measure to prefer. A body of work has questioned whether automated coherence measures are valid at all. Hoyle et al. (2021) argued that automated topic model evaluation is broken, showing that coherence metrics do not reliably reproduce human judgements of topic quality, and follow-up work extended the critique to neural topic models specifically (Hoyle et al., 2022) **[REFERENCE TO VERIFY — both Hoyle entries cited here and in Table 2.1, neither in the reference list]**. More recent analysis published in *Computational Linguistics* found that NPMI and C_v overstate differences between models relative to human assessment, with false discovery rates between 16% and 56% depending on the evaluation task, and identified a negative correlation between a topic's automated coherence score and how familiar respondents were with its words (Doogan-Poet Sirakoff & Buntine, 2024) **[REFERENCE TO VERIFY — absent from the reference list; the 16–56% figures are quoted again in Chapter 5 §5.2.4, so this citation is load-bearing]**. In other words, the metrics can reward topics that humans find obscure. Alternative formulations continue to be proposed, including contextualised coherence metrics using embeddings instead of co-occurrence counts **[REFERENCE TO VERIFY — bare identifier arXiv:2305.14587]**, and use-oriented evaluations that measure whether a topic model supports the task it was built for.

This literature has a direct bearing on the present study. If C_v and NPMI are unreliable proxies for interpretability, then reporting a single coherence figure would be a weak basis for a claim about taxonomy quality. The approach taken here — embedding-based coherence supplemented by category balance, platform spread, unclassified rate and manual inspection — follows the direction this critique points in, which is towards several imperfect indicators read together with a human check, and away from a single automated score presented as decisive.

For RAG the field is younger still, and Yu et al. (2024) identify the absence of diagnostic, component-level evaluation as a live problem: aggregate scores cannot tell a developer whether a failure originated in retrieval or in generation.

Table 2.1 compares representative studies on the dimensions that determine how far their conclusions transfer.

**Table 2.1** — Comparison of representative studies across datasets, methods and evaluation

| Study | Domain(s) | Methods compared | Metrics reported | Principal limitation |
|---|---|---|---|---|
| Turan et al. (2024) | Amazon reviews | LDA, NMF, BERTopic | Topic interpretability | Single platform; qualitative comparison |
| Mishra (2026) | Marketing research abstracts | BERTopic, LDA | Coherence, interpretability | Not customer feedback; small corpus |
| *Electronics* (2024) **[REF TO VERIFY]** | Amazon products | ML, DL, transformers | Accuracy, F1 | Single platform; aggregate metrics only |
| Zhang et al. (2023) | Multi-domain reviews | Feature projection, multi-source attention | Accuracy | Purpose-built architecture, not off-the-shelf comparison |
| Chen et al. (2024) | Wikipedia-based QA | LLMs under retrieval noise | Accuracy under perturbation | Not customer feedback; synthetic noise |
| Hoyle et al. (2021, 2022) | Multiple corpora | Topic models vs human judgement | Human agreement vs NPMI, C_v | Methodological critique, no applied pipeline |
| **This study** | **Amazon, Yelp, Twitter** | **Lexicon, classical ML, pretrained and fine-tuned transformers; RAG vs standalone LLM** | **Macro and per-class F1, confusion matrices, per-platform; embedding coherence, coverage; four RAGAS metrics** | **Rating-derived labels; single judge model** |

**Strengths and weaknesses.** Table 2.2 summarises how the three method families compare on the dimensions that matter for a practitioner choosing between them.

**Table 2.2** — Comparison of method families across the three stages of feedback analysis

| Dimension | Lexicon-based | Classical ML | Transformer | RAG (generation stage) |
|---|---|---|---|---|
| Training data required | None | Moderate labelled set | Large labelled set for fine-tuning | None for retrieval; relies on a pretrained generator |
| Compute cost | Negligible | Low | High (GPU for training) | Moderate; inference plus index |
| Handles context and word order | No | Weakly | Yes | Yes |
| Domain adaptability | Poor without manual lexicon work | Retrain on target data | Fine-tune on target data | Swap the underlying corpus |
| Interpretability | High: rules are inspectable | Moderate: feature weights | Low | Moderate: retrieved sources are inspectable |
| Typical evaluation | Accuracy, F1 | Accuracy, F1 | Accuracy, macro F1 | Faithfulness, answer relevancy, context precision and recall |
| Principal weakness | Misses sarcasm and implicit sentiment | Discards word order | Cost, and sensitivity to domain shift | Grounding is not guaranteed by architecture alone |

**Research trends.** Three are visible. Evaluation is shifting from reference-based to reference-free, driven by the cost of annotation and the difficulty of defining a single correct generated answer. Attention is moving from model architecture towards data quality and retrieval quality, on the recognition that a strong generator conditioned on poor context still fails. And explainability is moving from post-hoc explanation towards source attribution, where the justification for an output is the evidence retrieved to produce it.

### 2.1.4 Research gap

Four limitations run through the work reviewed above.

Comparisons in the studies reviewed above are conducted within a single domain, so the reported ranking of methods may reflect the dataset as much as the methods. A practitioner holding feedback from several sources has no direct evidence on which to base a decision.

Aggregate metrics dominate, so failure modes are hidden. Where a model fails on one class or one platform, the literature's reporting conventions tend to obscure it, and the mechanism is rarely diagnosed.

Topic model output is treated as though it were a taxonomy. Studies compare topic quality between algorithms, but the step from keyword clusters to a consolidated, named set of categories usable across domains receives little attention, and its validation without ground truth receives less.

RAG is frequently adopted without being compared against the baseline it is meant to improve on. Faithfulness is measured within RAG systems, but among the studies reviewed here the comparison against unconstrained generation — which a practitioner would weigh it against — did not appear, and the reliability of the language model doing the judging is only now being scrutinised.

This dissertation addresses these together. It compares method families at each pipeline stage on one multi-domain corpus with a shared protocol, reports per-class and per-platform behaviour alongside aggregates and diagnoses the mechanism behind results that contradict expectation, treats taxonomy consolidation as an explicit stage with its own unsupervised evaluation, and evaluates the retrieval-augmented system against a standalone language model baseline while stating openly the self-assessment bias in its judge.

---

## 2.2 Supporting Literature

### 2.2.1 Methodology

The design adopted here is a controlled comparative experiment built around a working pipeline. Comparative empirical studies in applied machine learning conventionally hold the data, the split and the metrics constant while varying the method, and that convention is followed. Where this study departs from much of the reviewed work is in comparing at each stage separately, since the methods under comparison perform different jobs and cannot be placed in a single table.

The literature also motivates a diagnostic rather than purely competitive stance. Because aggregate metrics conceal class-level failure, per-class F1 and confusion matrices are treated here as primary results, not appendix material.

### 2.2.2 Datasets

Three sources are combined. Amazon product reviews and Yelp business reviews are long-established benchmarks in this literature, both large and rating-labelled. The Twitter US Airline Sentiment dataset differs in an important way: its labels come from human annotation instead of star ratings, and its text is short and informal.

That difference is deliberate. Combining rating-derived and human-annotated labels introduces heterogeneity, which is a limitation, but it also produces the register variation the comparison depends on. A corpus of three Amazon-like sources would not test domain transfer at all.

Two dataset limitations follow from the literature and are carried forward into Chapter 3. Rating-derived labelling is approximate, as discussed in Section 2.1.3. And review corpora are subject to selection effects, since customers with strong opinions are more likely to write, so the sentiment distribution should not be read as representative of customers generally.

### 2.2.3 Evaluation methods

Three evaluation regimes are used, matched to what each stage permits.

For sentiment classification, labels exist, so standard supervised metrics apply. Macro F1 is preferred over accuracy as the headline figure because it weights all three classes equally and therefore exposes the single-class collapse that the reviewed literature tends to hide.

For complaint discovery there is no ground truth, so classification metrics are unavailable. The literature's usual proxies are coherence measures, and the disagreement between studies noted in Section 2.1.2 shows they are not decisive. This study uses embedding-based coherence — the similarity of reviews to their category centroid — supplemented by category balance, platform distribution, and the proportion of reviews left unclassified. Reporting several weak indicators together is more defensible than resting on one.

For the generation stage, RAGAS provides reference-free faithfulness, answer relevancy, context precision and context recall (Es et al., 2024). Four metrics are used, not one, because they can disagree in informative ways: an unconstrained model can score highly on relevancy while having no evidential basis at all, so relevancy alone would reward the behaviour the system is designed to prevent.

The LLM-as-judge literature bears directly on a constraint of this study. Because the same model both generates and judges, self-enhancement bias of the kind documented by Gu et al. (2024) and Li et al. (2024) is a genuine threat to validity. This was accepted as a consequence of free-tier API limits, and it is reported as a limitation instead of being left implicit; using an independent judge model is identified as future work.

---

## 2.3 Technology Review

### 2.3.1 Models and libraries

**Sentiment models.** VADER is retained as a baseline because it requires no training and its decisions are inspectable, which makes it a meaningful floor for what rules alone achieve (Hutto & Gilbert, 2014). Naive Bayes and logistic regression over TF-IDF features are implemented in scikit-learn; they are cheap, strong on longer text and, importantly, tunable to a standard comparable with the transformers. RoBERTa is used in its Twitter-pretrained form (Barbieri et al., 2020), which makes the domain-transfer question testable. DistilBERT is chosen for fine-tuning over BERT or RoBERTa because distillation retains most of the accuracy at around 40% fewer parameters (Sanh et al., 2019), which makes training feasible on freely available hardware — a real constraint for an MSc project, and one worth stating openly instead of disguising as a purely technical preference.

**Topic modelling.** BERTopic is selected over LDA on the evidence reviewed in Section 2.1.2, particularly its handling of short informal text, which matters because one of the three sources is tweets. Its components are used as intended: sentence-transformer embeddings (Reimers & Gurevych, 2019), UMAP for dimensionality reduction (McInnes et al., 2018), and HDBSCAN for clustering (Campello et al., 2013). HDBSCAN's ability to leave points unassigned as noise is an advantage here, since not every negative review expresses a categorisable complaint.

**Category assignment.** Reviews are assigned to discovered categories using zero-shot natural language inference, following the approach behind the DeBERTa-based universal classifiers (Laurer et al., 2023). This avoids needing labelled examples for categories that did not exist before the discovery stage ran. An embedding shortlist precedes the entailment step so that inference cost stays bounded as the taxonomy grows.

**Retrieval.** FAISS provides the vector index (Johnson et al., 2019). The embedding model is `all-MiniLM-L6-v2` from the sentence-transformers family (Reimers & Gurevych, 2019), chosen for its 384-dimensional output and speed; a larger embedding model would improve representation quality at a cost in index size and query latency, and that trade-off is examined empirically rather than assumed.

**Generation.** Groq's hosted Llama 3.1 model is used, orchestrated through LangChain. The choice was driven by cost — the free tier permitted the volume of evaluation calls required — and it carries consequences that are reported honestly: rate limits constrained sample sizes, and using the same provider for generation and judging introduced the bias discussed above.

**Evaluation and tracking.** RAGAS supplies the generation metrics (Es et al., 2024), scikit-learn the classification metrics, and MLflow records parameters, metrics and artefacts so that individual runs remain recoverable.

**Table 2.3** — Technology selection and justification

| Component | Selected | Alternatives considered | Reason for selection |
|---|---|---|---|
| Lexicon baseline | VADER | TextBlob, SentiWordNet | Designed for social media text; widely used, so results are comparable |
| Classical models | Naive Bayes, logistic regression (TF-IDF) | SVM, gradient boosting | Fast, interpretable, standard baselines in this literature |
| Pretrained transformer | `twitter-roberta-base-sentiment-latest` | BERT-base, generic RoBERTa | Twitter pretraining makes domain transfer directly testable |
| Fine-tuned transformer | DistilBERT | BERT, RoBERTa | ~40% fewer parameters at comparable accuracy; trainable on available hardware |
| Topic model | BERTopic | LDA, NMF, Top2Vec | Handles short informal text; semantic clustering rather than lexical |
| Embeddings | `all-MiniLM-L6-v2` | `all-mpnet-base-v2`, OpenAI embeddings | Fast, 384-d, no API cost; quality trade-off tested empirically |
| Vector index | FAISS | Chroma, Pinecone, Weaviate | Local, free, well documented, no external dependency |
| Zero-shot classifier | DeBERTa-v3 zero-shot | `bart-large-mnli` | Stronger reported zero-shot performance with larger label sets |
| Generator | Llama 3.1 via Groq | GPT-4, Gemini, local Llama | Free tier supported the required evaluation volume |
| Orchestration | LangChain | LlamaIndex, custom | Mature retrieval-chain abstractions; large ecosystem |
| RAG evaluation | RAGAS | TruLens, ARES, manual | Reference-free metrics; no annotation budget required |

### 2.3.2 Trade-offs acknowledged

Two choices in Table 2.3 deserve explicit qualification. Selecting a small embedding model and a free-tier generator were both constrained decisions, not optimal ones. Their effect is examined directly: retrieval quality is compared against a TF-IDF baseline, and the limits imposed by rate limiting on evaluation sample sizes are reported in Chapter 5 rather than glossed over.

---

## 2.4 Chapter Summary

This review establishes four things. Fine-tuned transformers are the dominant approach to sentiment classification, but the evidence supporting their superiority comes overwhelmingly from single-domain studies reporting aggregate metrics, so it is weaker than it appears. BERTopic is better suited than LDA to the short, informal text found in social feedback, though comparative results are mixed and depend on the dataset and the coherence measure chosen. Retrieval-augmented generation is the accepted response to hallucination, but adopting it does not by itself establish that grounding has been achieved, and comparisons against unconstrained generation remain uncommon. Finally, reference-free evaluation using a language model as judge is now standard practice while its reliability is still being actively questioned.

The technologies selected follow from this. VADER as a floor, tuned classical models as a serious baseline, a Twitter-pretrained RoBERTa to make domain transfer testable, a distilled transformer for feasible fine-tuning, BERTopic with its embedding and clustering components, zero-shot entailment for assigning reviews to newly discovered categories, and FAISS with RAGAS for the retrieval and evaluation stages.

The gap that remains is the absence of controlled, stage-wise comparison on multi-domain data, reported at a granularity that reveals failure modes instead of concealing them, with the generated output verified against the evidence it claims to rest on.

Chapter 3 sets out the methodology designed to close that gap. It describes the research approach and its phases, the datasets and their preparation, the experimental design that keeps the comparison controlled, the models and hyperparameters used at each stage, the metrics chosen and why, and the ethical considerations governing the data. Each decision there traces back to a limitation identified in this chapter: the shared split responds to the confounding in single-dataset comparisons, per-class reporting responds to the aggregate-metric problem, the unsupervised taxonomy evaluation responds to the absence of ground truth, and the standalone baseline responds to the missing comparison in the RAG literature.
