# Chapter 1 — Introduction

This dissertation asks a question practitioners face directly: which method is actually justified for which part of customer feedback analysis? The work is a comparative study. A pipeline is built, because the methods cannot be compared without one, but the pipeline is the instrument and the comparison is the result.

Feedback analysis is not one task but three, and different methods compete at each: sentiment classification is supervised, complaint discovery is unsupervised, and answering an analyst's question sets retrieval-augmented generation against a language model working alone. The comparison is therefore made stage by stage, with competing methods tested on the same data under the same protocol against an established baseline. A single table spanning all three would be meaningless, since the methods do different jobs and are judged by different measures.

All experiments run on 642,692 reviews from three deliberately dissimilar sources: Amazon product reviews, Yelp business reviews, and customer tweets to airlines. Customer feedback is one of the few sources of honest, unsolicited information a business holds about its own failures, and it arrives faster than any team can read it. Automation is attractive, and the choice of method is usually made on reputation rather than evidence — newer is assumed better. This study tests that assumption on data resembling what an organisation would actually hold, and reports the several places it does not hold.

Four findings run against the expected ordering, and they are why the study is worth reading. A pretrained transformer widely regarded as strong on social text scores below logistic regression on the pooled corpus. Topic modelling produces a taxonomy in which more than a quarter of sampled reviews fall into a catch-all category. Retrieval-augmented generation scores *lower* than a plain language model on answer relevancy, while being the only one of the two whose answers can be checked. And the one design decision tested in isolation — tagging retrieved reviews with their platform and rating — buys grounding at a measurable cost to how responsive the answers read, rather than being the free improvement the design reasoning assumed.

---

## 1.1 Background

### Context and scope

The work sits in applied data science, drawing on natural language processing, information retrieval and large language models. The tools are now accessible enough that one developer can assemble in weeks what once needed a research team, and that accessibility creates the problem addressed here: when every method is easy to run, choosing between them becomes the hard part, and the published evidence for that choice is thinner than it appears. Sentiment dashboards report proportions; they rarely explain causes and almost never show their evidence.

The study addresses one question at three stages. How do classical methods compare with transformers on feedback from several domains, and is fine-tuning worth its cost? Can topic modelling turn heterogeneous reviews into a taxonomy that holds across platforms, and how is that judged when no correct answer exists? Does retrieval-augmented generation offer a measurable advantage over a language model alone?

Four boundaries apply. Three platforms are examined, so the findings speak to cross-domain behaviour rather than to customer feedback in general. Only English text is analysed. Complaint discovery uses negative reviews only. And the generation stage is evaluated on five questions — enough to demonstrate the measurement approach and expose a directional pattern, not enough to support a confident quantitative claim. That constraint is stated here rather than buried in Chapter 5, because it bounds what RQ2 can conclude.

### The state of the art and its limits

Sentiment analysis is the most established part of the picture. Lexicon tools such as VADER score text using a hand-built dictionary plus rules for negation and emphasis (Hutto and Gilbert, 2014). They need no training data and their decisions can be inspected, which is why VADER is treated here as a baseline in its own right rather than a member of the classical family: it learns nothing from the data, so it measures how far fixed rules go before any training is involved. Its weakness is judging words largely in isolation, so sarcasm and implied criticism are missed.

Supervised classical models learn from labelled examples. Naive Bayes and logistic regression over TF-IDF features are cheap, fast and surprisingly strong, though they discard word order (Pang and Lee, 2008). Transformers keep it: pretrained encoders such as RoBERTa represent each word in context, and fine-tuning a smaller model such as DistilBERT adapts it further (Devlin et al., 2019; Sanh et al., 2019; Liu et al., 2019).

The usual assumption is that this ordering also ranks performance. It does not always hold, and the reason is partly methodological. Published comparisons are typically run on a single dataset, often a balanced benchmark of product reviews. A model scoring well there may do poorly on tweets. Studies testing the same models across several feedback types at once, on one corpus and one protocol, did not appear in the literature reviewed for this project. That leaves a practitioner without an answer to a simple question: is fine-tuning worth its cost on my data, or would a linear model serve?

Knowing sentiment is only half of what a business needs. It says how much dissatisfaction exists, not what it concerns. Latent Dirichlet Allocation, long the standard, treats each document as a mixture of word-based topics (Blei et al., 2003) and works poorly on short informal text, because it relies on words co-occurring within a document and tweets supply too few. BERTopic handles such text better, embedding each document with a sentence transformer, reducing dimensions, clustering, then describing each cluster by its distinctive words (Grootendorst, 2022). Because it groups by meaning rather than shared vocabulary, two differently worded complaints can still land together.

What BERTopic returns, though, is a list of keyword clusters, not a set of actionable complaint categories. The gap is larger than it looks. Running BERTopic per platform here produced 69 topics for Amazon, 18 for Yelp and 34 for Twitter Airline — 121 in total. Many describe the same underlying issue in different words; some reflect quirks of the source data; the level of detail varies sharply between platforms. Consolidating these needs a further step, and here the literature is thin. There is no agreed way to decide how many categories to keep, or to validate the result when no correct answer exists.

The final part concerns how people reach the analysis. Managers ask specific questions in ordinary language. Language models suit that, but a model asked about a collection it has never seen will sometimes invent figures, categories or sources, and will do so fluently (Ji et al., 2023). Retrieval-augmented generation is the standard remedy: relevant documents are fetched first, and the model is instructed to answer only from those (Lewis et al., 2020). Adopting RAG does not guarantee grounding — retrieved text can be off-target, and the model can ignore it. Whether grounding has been achieved must be measured, and it is measured far less often than RAG is adopted. Direct comparison against a plain baseline was rarer still in the literature reviewed here, even though that is the baseline any practitioner would weigh RAG against.

Two gaps follow: the shortage of controlled comparison at each stage on the same multi-domain data, and the shortage of verification of what these systems produce. Both matter because the cost falls on whoever uses the results. A wrong complaint category sends effort in the wrong direction. An unsupported claim in a generated summary is worse, because it looks exactly like a supported one.

### Research questions

Four questions follow, each answerable with evidence produced here:

- **RQ1.** How does sentiment classification performance differ between lexicon-based, classical machine learning, pretrained transformer and fine-tuned transformer approaches on multi-domain customer feedback?
- **RQ2.** How effectively does retrieval-augmented generation improve complaint exploration compared with a standalone large language model?
- **RQ3.** Can BERTopic produce a coherent cross-domain complaint taxonomy from heterogeneous customer reviews?
- **RQ4.** To what extent can retrieval-augmented answers be traced to, and verified against, the source reviews used to produce them?

RQ4 is framed around traceability rather than business usefulness. Establishing usefulness would require a user study, and none was carried out. Traceability can be measured directly, and is.

---

## 1.2 Research Aim and Objectives

### Aim

To compare, stage by stage, the classical machine learning, transformer-based and retrieval-augmented approaches available at each point in a customer complaint analysis pipeline, each measured against an established baseline on a single multi-domain corpus, in order to establish which approach is justified at which stage and how far the output of each can be trusted.

The artefact produced is the pipeline the comparison requires: a sentiment classification component covering the baseline and both learned families, a complaint discovery and categorisation component built on BERTopic, a FAISS index supporting semantic retrieval, and a question-answering component returning source review identifiers with every answer. It is intended for customer experience and product analysts who need to understand recurring complaints without reading every review, and for practitioners choosing between these methods.

The aim is realistic within the timeframe because it uses existing models rather than training new architectures. The one model trained here is a fine-tuned DistilBERT classifier, small enough for freely available hardware. The heavier work lies in the evaluation, which is where the contribution sits.

### Objectives

1. **Review the literature** on sentiment analysis, topic modelling, retrieval-augmented generation and explainable AI, identifying where existing comparisons are absent or restricted to a single domain. Judged on whether the review compares methods and justifies those later chosen, rather than describing them in sequence. *Delivered in Chapter 2.*

2. **Prepare a unified multi-domain dataset** suitable for controlled comparison: one cleaned corpus of 642,692 reviews from three platforms sharing a schema, with sentiment labels from a documented rule, class balance recorded, and a fixed seed so the split can be recreated. A shared split is what makes the comparison controlled for four of the five models; the fine-tuned model is the exception, disclosed in Section 3.5.1.

3. **Implement and compare five sentiment classifiers** spanning a lexicon baseline, the classical family, and pretrained and fine-tuned transformers — and explain the mechanism behind any result running against expectation. Requires accuracy, macro F1, per-class F1 and confusion matrices on one held-out set of 128,539 reviews, with a per-platform breakdown. A ranking alone would leave the most interesting finding unexplained, so diagnosis forms part of the objective. *Addresses RQ1.*

4. **Discover complaint topics per platform using BERTopic and consolidate them** into one cross-domain taxonomy: 121 raw topics reduced to 24 named categories, with the merging procedure and parameters recorded, and coverage and coherence measured rather than assumed. Discovery runs per platform because pooling would let the largest platform dominate. *Addresses RQ3.*

5. **Build the retrieval-augmented question-answering component and a standalone baseline.** Requires documented embedding model, index configuration, retrieval depth and prompt design, returning source identifiers with each answer. Retrieved reviews must carry platform, rating and sentiment into the prompt, since otherwise the instruction to cite only platforms present in the evidence cannot be enforced. *Addresses RQ2 and RQ4.*

6. **Evaluate every stage using stage-appropriate measures**, comparing against baselines, ablating design decisions where feasible, and diagnosing anomalous results. Supervised metrics for classification; coherence and distribution measures for the taxonomy; RAGAS faithfulness, answer relevancy, context precision and context recall for both generation conditions. Three diagnostic experiments test the proposed explanation of the anomaly: a per-platform breakdown, a re-evaluation without the neutral class, and a decision-threshold sweep. One retrieval design decision — metadata tagging — is ablated directly, so at least one is measured rather than argued. All results are written to file. *Addresses RQ1–RQ4.*

7. **Discuss the findings, judge whether the aim was met, and state the limitations openly.** Success means each research question receives a direct answer supported by reported evidence, including unfavourable results. *Delivered in Chapters 5 and 6.*

### Contributions

Stated at the level of a careful applied comparison, deliberately:

1. **A controlled multi-domain comparison of five sentiment classifiers** on one corpus, one protocol and one held-out set of 128,539 reviews, with per-class and per-platform breakdowns rather than a single aggregate.
2. **A diagnostic explanation of an anomalous result** — a pretrained transformer below a linear baseline — supported by three follow-up experiments rather than asserted.
3. **An end-to-end complaint taxonomy pipeline**, from per-platform discovery through cross-platform merging to threshold-gated assignment, with its coverage failures reported rather than smoothed over.
4. **A direct RAG-versus-standalone comparison** on the same questions with the same generator, measuring grounding rather than assuming it, and reporting the case where retrieval *reduced* a quality metric.
5. **An executed ablation of the metadata-tagging decision**, which returned a trade-off between grounding and responsiveness rather than the predicted improvement, together with one directly observed hallucination that tagging prevented.
6. **A reproducible evaluation harness**: every reported figure is produced by a script writing to `data/results/`, so numerical claims are auditable rather than transcribed.

Not claimed: no new algorithm or architecture, no user study establishing business usefulness, and a generation evaluation too small for confident quantitative conclusions. These boundaries are restated in Chapter 6.

---

## 1.3 Research Approach

The approach is quantitative and experimental, organised so that methods competing at the same stage are tested under identical conditions — a comparison is only as good as the controls around it. Each stage is evaluated before the next depends on it.

The three sources are mapped onto a common schema, cleaned, and labelled from star ratings: two or below negative, three neutral, four or above positive. Twitter Airline carries its own annotation, mapped to the same classes. The rule is simple and repeatable but approximate, since a four-star review can still contain a complaint. This places a ceiling on the accuracy any classifier here can reach, and reading the results without knowing it would be misleading.

All five classifiers are evaluated on one stratified split with a fixed seed, so for four of them differences cannot come from differences in test data. Metrics and parameters are logged to MLflow, keeping the comparison auditable.

BERTopic runs on the negative reviews of each platform separately; topics are merged across platforms by clustering centroid vectors, and each merged group is named by a language model. Reviews are assigned in two steps — an embedding comparison shortlists likely categories, then a zero-shot entailment model ranks the shortlist. A confidence threshold leaves weak matches unclassified rather than forcing a label, keeping a known unknown visible. As Chapter 4 shows, that unknown proves large enough to be a finding in its own right.

The corpus is indexed with a sentence-transformer model in FAISS. Questions retrieve a small number of reviews, which are passed to the language model under instructions to answer only from that text; each is tagged with platform, rating and sentiment first, and identifiers are returned with every answer. The same questions are then put to the model with no retrieval, providing the RQ2 baseline.

Evaluation uses three regimes, chosen by what each stage allows. Classification has labels, so standard metrics apply, with McNemar's exact test for pairwise comparison. The taxonomy has none, so quality is argued from how tightly reviews cluster around their category, how evenly categories are populated, how they spread across platforms, and how many fall below the threshold. Question answering is scored with RAGAS. Two limitations belong on record at the outset: the evaluation covers five questions, and the model judging answers is the model producing them, introducing self-assessment bias. Both follow from free-tier API limits, and both are reported openly rather than allowed to inflate the generation results.

Tools are Python with pandas and scikit-learn, Hugging Face Transformers for RoBERTa and DistilBERT, VADER for the lexicon baseline, BERTopic with UMAP and HDBSCAN for discovery, sentence-transformers for embeddings, FAISS for the index, LangChain for the retrieval chain, Groq's hosted Llama 3.1 for generation, RAGAS for generation metrics, and MLflow for tracking. Seeds are fixed, dependency versions pinned, and unit tests cover the topic-merging and category-shortlisting logic.

On ethics, all three datasets are public and were collected by their original publishers, so no new data was gathered and no participants involved. Reviews contain no special-category personal data, reviewer names are unused, and reviews are referenced by identifier only. Two risks are recorded: review text occasionally contains personal detail written by the reviewer, so no attempt is made to link reviews to individuals; and generated summaries describe named businesses, where acting on an unsupported summary could affect one unfairly. Returning source reviews with every answer is a partial safeguard, since a claim can be checked before it is used.

---

## 1.4 Dissertation Outline

**Chapter 2** reviews work on sentiment classification, topic modelling, dense retrieval and retrieval-augmented generation, comparing approaches rather than listing them, and identifies the two gaps this study addresses.

**Chapter 3** sets out the methodology: datasets and preprocessing, the experimental design that makes the comparison controlled, models and hyperparameters, and the reasoning behind each choice including alternatives rejected.

**Chapter 4** reports the analytical work and results: exploratory analysis, the sentiment experiments and their diagnostic follow-ups, topic discovery and taxonomy construction, and the retrieval, generation and ablation evaluations.

**Chapter 5** interprets those results, answers each research question, compares the findings with Chapter 2's literature, and examines the results running against expectation. It also works through a case where tracing an unexpected zero back to its source changed the diagnosis: a question the pipeline never answered turned out to have been rejected by a scope classifier rather than declined for want of evidence.

**Chapter 6** reviews the aim and objectives, states what was and was not established, and sets out future work drawn from the limitations actually encountered.
