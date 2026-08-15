# Chapter 1 — Introduction

**Intelligent Customer Feedback Analytics using LLMs**

*A stage-wise empirical comparison of classical machine learning, transformer-based models, and retrieval-augmented generation for customer complaint analysis and feedback exploration.*

---

This dissertation is a comparative study. It asks a question practitioners face directly: which approach is actually justified for which part of the work? Analysing customer feedback is not one task but several, and the methods that compete at each point are different. Classifying sentiment is a supervised problem where classical machine learning and transformer models are the realistic candidates. Identifying what customers complain about is unsupervised, and belongs to topic modelling. Answering an analyst's questions about those complaints pits retrieval-augmented generation against a language model working alone.

The comparison is therefore made stage by stage. Within each stage, competing methods are tested on the same data under the same protocol, and each is measured against an established baseline. This is a deliberate design choice rather than a compromise. A single head-to-head table across all three method families would be meaningless, because they perform different jobs and are judged by different measures.

The overall aim is to produce evidence, not a demonstration. A pipeline is built, because the methods cannot be compared without one, but the pipeline is the instrument and the comparison is the result. All experiments run on 642,692 reviews drawn from three very different sources: Amazon product reviews, Yelp business reviews, and customer tweets directed at airlines.

The topic matters for a practical reason. Customer feedback is one of the few sources of honest, unsolicited information a business holds about its own failures, and it arrives faster than any team can read it. Automation is therefore attractive, and the choice of method is usually made on reputation rather than evidence. Newer approaches are assumed to be better. This dissertation tests that assumption on data resembling what an organisation would actually hold, and reports the several places where it does not hold.

Three of the findings run against the expected ordering, and they are the reason the study is worth reading rather than an embarrassment to be managed. A pretrained transformer widely regarded as strong on social text scores below a logistic regression baseline on the pooled corpus. Topic modelling produces a taxonomy in which more than a quarter of sampled reviews fall into a catch-all category. Retrieval-augmented generation scores *lower* than a plain language model on answer relevancy, while being the only one of the two whose answers can be checked at all. Each of these is reported in full, explained where the evidence supports an explanation, and left open where it does not.

---

## 1.1 Background

### Context

The work sits in applied data science, drawing on natural language processing, information retrieval and large language models. All three fields have moved quickly, and the tools are now accessible enough that one developer can assemble in weeks what would once have needed a research team. That accessibility creates the problem this study addresses. When every method is easy to run, choosing between them becomes the hard part, and the published evidence for making that choice is thinner than it appears.

The area matters because unstructured text makes up most of the data organisations hold and remains the hardest to use. Structured data has mature tooling. Feedback text does not, at least not in a form that answers the questions managers ask. Sentiment dashboards report proportions. They rarely explain causes, and they almost never show their evidence.

Within that field this project belongs to the empirical end. It compares existing methods under controlled conditions; it does not propose a new algorithm. The contribution is the comparison itself, together with the evaluation design that makes it fair and the diagnostic work that explains the anomalies it uncovers.

### Scope

The study addresses one question at three stages. At the classification stage, how do classical machine learning methods compare with transformer-based models on feedback drawn from several different domains, and is fine-tuning worth its cost? At the discovery stage, can topic modelling turn heterogeneous reviews into a complaint taxonomy that holds across platforms, and how is that judged when no correct answer exists? At the exploration stage, does retrieval-augmented generation offer a measurable advantage over a language model used on its own?

Four boundaries apply. Three platforms are examined, so the findings speak to behaviour across domains but not to customer feedback in general. Only English-language text is analysed. Complaint discovery uses negative reviews only, since the object of interest is a set of complaint categories rather than a general description of the corpus. Finally, the generation stage is evaluated on five questions, which is enough to demonstrate the measurement approach and to expose a clear directional pattern, but not enough to support a confident quantitative claim about generation quality. That constraint is stated here rather than buried in Chapter 5, because it bounds what RQ2 can conclude.

### Current state of the art and its limits

Sentiment analysis is the most established part of the picture, and its history is worth following, because each stage solved a problem and introduced another. Lexicon-based tools such as VADER score text using a hand-built dictionary plus rules for negation and emphasis (Hutto & Gilbert, 2014). They need no training data and their decisions can be inspected. That combination is why VADER is treated here as a baseline in its own right and not as a member of the classical machine learning family: it learns nothing from the data, so it measures how far a fixed set of rules can go before any training is involved. Every other method in this study should be expected to beat it, and the size of that margin is itself informative. Its weakness is that it judges words largely in isolation, so sarcasm and implied criticism are often missed.

Supervised classical models learn from labelled examples instead. Naive Bayes and logistic regression over TF-IDF features are cheap, fast and surprisingly strong, though they discard word order (Pang & Lee, 2008). Transformers keep it. Pretrained encoders such as RoBERTa represent each word in the light of its context, and fine-tuning a smaller model such as DistilBERT on data from the target domain adapts it further (Devlin et al., 2019; Sanh et al., 2019; Liu et al., 2019).

The usual assumption is that this ordering also ranks performance. It does not always hold, and the reason is partly methodological. Published comparisons are typically run on a single dataset, often a balanced benchmark of product or film reviews. A model that scores well there may do poorly on tweets. Studies that test the same models across several types of feedback at once, on one corpus and one protocol, did not appear in the literature reviewed for this project (Chapter 2). That leaves a practitioner without an answer to a simple question: is fine-tuning worth its cost on my data, or would a linear model serve?

Knowing the sentiment of a review is only half of what a business needs. It says how much dissatisfaction exists, not what it concerns. Topic modelling is the usual next step. Latent Dirichlet Allocation, long the standard, treats each document as a mixture of word-based topics (Blei et al., 2003). It works poorly on short informal text, because it relies on words co-occurring within a document and tweets supply too few. BERTopic handles such text better. It embeds each document with a sentence transformer, reduces the dimensions, clusters the result, and describes each cluster using distinctive words (Grootendorst, 2022). Because it groups by meaning instead of shared vocabulary, two complaints worded differently can still land together.

What BERTopic returns, though, is a list of keyword clusters, and that is not a set of complaint categories a business could act on. The gap is larger than it looks. Running BERTopic separately on each platform in this project produced 69 topics for Amazon, 18 for Yelp and 34 for Twitter Airline, giving 121 in total. Many describe the same underlying issue in different words. Some reflect quirks of the source data instead of real complaints. The level of detail also varies sharply between platforms. Consolidating these into one taxonomy needs a further step, and here the literature is thin. There is no agreed way to decide how many categories to keep, or to check the result when no correct answer exists for comparison. This project's own taxonomy inherits that difficulty rather than escaping it, as Chapter 4 reports.

The final part concerns how people reach the analysis. Managers ask specific questions in ordinary language. Large language models suit that style of interaction, but a model asked about a collection it has never seen will sometimes invent figures, categories or sources, and it will do so fluently (Ji et al., 2023). Retrieval-Augmented Generation is the standard remedy: relevant documents are fetched from an index first, and the model is instructed to answer only from those documents (Lewis et al., 2020). Adopting RAG does not guarantee grounding. Retrieved text can be off-target, and the model can ignore it. Whether grounding has been achieved must be measured, and it is measured far less often than RAG is adopted. Direct comparisons against a plain language-model baseline were rarer still in the literature reviewed here, even though that is the baseline any practitioner would weigh RAG against.

Two gaps follow. The first is the shortage of controlled comparison at each stage on the same multi-domain data, which leaves method selection resting on reputation. The second is the shortage of verification of what these systems produce, particularly for generated output. Both matter because the cost falls on whoever uses the results. A wrong complaint category sends effort in the wrong direction. An unsupported claim in a generated summary is worse, because it looks exactly like a supported one.

### Motivation

The reason for addressing this is straightforward. Feedback analysis is already automated in industry and its results already inform decisions, yet the choice of method is often made without evidence from data resembling the organisation's own. A comparison run across three domains, with one corpus, one protocol and one held-out test set, gives a firmer basis for that choice than a benchmark score quoted from a paper.

The expected benefits are practical. A practitioner gains evidence on which method family is worth its cost for which task, including the cases found here where the cheaper option is sufficient and the more sophisticated one is not. An analyst gains complaint categories derived from data instead of assumption, together with an honest account of where those categories break down. A researcher gains a documented account of how these components behave together, including where the combination performs worse than expected. The beneficiaries are customer experience and product teams who currently read samples of feedback by hand, and applied researchers building similar systems who would benefit from knowing which parts are fragile.

### The research problem

Existing work supplies methods but not the comparative evidence needed to choose between them. Sentiment models are benchmarked on single datasets, so their reported accuracy may not survive a change of domain. Topic models produce clusters that are not yet categories, with no accepted way to validate them without labels. Language models generate confident text whether or not evidence supports it, and retrieval-augmented systems are frequently adopted without being tested against the simpler alternative they are meant to improve on. Traceability, which is what allows a claim to be checked, tends to be treated as a matter of presentation when it is really a matter of design.

Four questions follow, each answerable with evidence produced in this project:

- **RQ1.** How does sentiment classification performance differ between lexicon-based, classical machine learning, pretrained transformer and fine-tuned transformer approaches on multi-domain customer feedback?
- **RQ2.** How effectively does Retrieval-Augmented Generation improve complaint exploration compared with a standalone large language model?
- **RQ3.** Can BERTopic produce a coherent cross-domain complaint taxonomy from heterogeneous customer reviews?
- **RQ4.** To what extent can retrieval-augmented answers be traced to, and verified against, the source reviews used to produce them?

RQ4 is framed around traceability instead of business usefulness. Establishing usefulness would require a user study, and none was carried out. Traceability can be measured directly, and is.

---

## 1.2 Research Aim and Objectives

### Aim

The aim of this research is to compare, stage by stage, the classical machine learning, transformer-based and retrieval-augmented approaches available at each point in a customer complaint analysis pipeline, each measured against an established baseline on a single multi-domain corpus, in order to establish which approach is justified at which stage and how far the output of each can be trusted.

The comparison requires a working pipeline, and that pipeline is the artefact produced. It has four connected parts: a sentiment classification component covering the baseline and both learned method families, a complaint discovery and categorisation component built on BERTopic, a FAISS index supporting semantic retrieval over the review corpus, and a question-answering component that returns source review identifiers with every answer. It is intended for customer experience and product analysts who need to understand recurring complaints without reading the underlying reviews in full, and for practitioners choosing between these methods for similar work.

The aim is realistic within the project timeframe because it uses existing models instead of training new architectures from scratch. The one model trained here is a fine-tuned DistilBERT classifier, small enough to train on freely available hardware. The heavier work lies in the evaluation, which is where the contribution sits.

### Objectives

**Objective 1 — Review the literature on sentiment analysis, topic modelling, retrieval-augmented generation and explainable AI, and identify where existing comparisons are absent or restricted to a single domain.** Judged on whether the review compares methods and justifies those later chosen, instead of describing them one after another. Delivered in Chapter 2.

**Objective 2 — Select a research methodology and prepare a unified multi-domain dataset suitable for controlled comparison.** The outcome is a single cleaned dataset of 642,692 reviews from three platforms sharing one schema, with sentiment labels produced by a documented rule, class balance recorded, and a fixed random seed so the same train and test split can be recreated. A shared dataset and split is what makes the comparison controlled for four of the five models: differences between them cannot then be attributed to differences in data. The fine-tuned model is the exception — it was developed against a separately generated corpus version, a limitation disclosed and discussed in Section 3.5.3 rather than assumed away.

**Objective 3 — Implement and compare five sentiment classifiers, spanning a lexicon-based baseline, the classical machine learning family, and pretrained and fine-tuned transformers, and explain the mechanism behind any result that runs against expectation.** Requires accuracy, macro F1, per-class F1 and confusion matrices for all five models on one held-out set of 128,539 reviews, together with a breakdown by platform. Reporting the baseline separately matters, since it sets the floor every learned method is judged against. A ranking alone would leave the most interesting finding unexplained, so the diagnostic explanation forms part of the objective rather than an optional extra. Addresses RQ1.

**Objective 4 — Discover complaint topics for each platform using BERTopic and consolidate them into one cross-domain taxonomy.** The outcome is a documented set of per-platform topics reduced from 121 raw topics to 24 named categories, with the merging procedure and its parameters recorded, and with the coverage and coherence of the result measured rather than assumed. Discovery runs per platform instead of over the pooled corpus, because pooling would let the largest platform dominate the topic space. Addresses RQ3.

**Objective 5 — Build the retrieval-augmented question-answering component and a standalone language model baseline to compare it against.** Requires a working pipeline with documented embedding model, index configuration, retrieval depth and prompt design, returning source identifiers with each answer. Retrieved reviews must carry their platform, rating and sentiment into the prompt, since otherwise the instruction to cite only platforms present in the evidence cannot be enforced. Addresses RQ2 and RQ4.

**Objective 6 — Evaluate every stage using measures appropriate to that stage, comparing against baselines and diagnosing the mechanism behind any anomalous result.** The outcome is supervised metrics for sentiment classification, coherence and distribution measures for the taxonomy, and RAGAS scores for faithfulness, answer relevancy, context precision and context recall for both the retrieval-augmented and standalone conditions. Where a result contradicts expectation, three further diagnostic experiments test the proposed explanation: a per-platform breakdown, a re-evaluation with the neutral class removed, and a decision-threshold sweep. All results are written to file so that every number in Chapters 4 and 5 can be traced to its source. Addresses RQ1 to RQ4.

**Objective 7 — Discuss the findings, judge whether the aim was met, and state the limitations openly.** Success means each research question receives a direct answer supported by reported evidence, including the unfavourable results, and that constraints such as rating-derived labels, the small generation-evaluation sample and the absence of a user study are discussed rather than minimised. Delivered in Chapters 5 and 6.

### Contributions

The contributions are those of a careful applied comparison, and are stated at that level deliberately:

1. **A controlled multi-domain comparison of five sentiment classifiers** on one corpus, one protocol and one held-out set of 128,539 reviews, with per-class and per-platform breakdowns rather than a single aggregate score.
2. **A diagnostic explanation of an anomalous result** — a pretrained transformer scoring below a linear baseline — supported by three follow-up experiments rather than asserted from intuition.
3. **An end-to-end complaint taxonomy pipeline** from per-platform topic discovery through cross-platform merging to threshold-gated category assignment, with its coverage failures reported rather than smoothed over.
4. **A direct RAG-versus-standalone-LLM comparison** on the same questions with the same generator, measuring grounding rather than assuming it, and reporting the case where retrieval *reduced* a quality metric.
5. **A reproducible evaluation harness**: every reported figure is produced by a script that writes to `data/results/`, so the numerical claims in this dissertation are auditable rather than transcribed.

What is *not* claimed: no new algorithm or architecture is proposed, no user study establishes business usefulness, and the generation evaluation is too small to support confident quantitative conclusions. These boundaries are set out again in Chapter 6.

---

## 1.3 Research Approach

The approach is quantitative and experimental. Its organising principle is that methods competing at the same stage are tested under identical conditions, since a comparison is only as good as the controls around it. Each stage is evaluated before the next depends on it.

Objective 2 is met by mapping all three sources onto a common schema, cleaning the text, and labelling sentiment from star ratings, with two stars or below treated as negative, three as neutral, and four or above as positive. Twitter Airline data carries its own sentiment annotation and is mapped onto the same three classes. The rule is simple and repeatable, but it is an approximation, because a four-star review can still contain a complaint. This is stated here because it places a ceiling on the accuracy any classifier in this study can reach, and reading the results without knowing it would be misleading.

Objective 3 is met by evaluating all five classifiers on one stratified split with a fixed seed. For four of the five, this means differences between models cannot come from differences in test data; the fine-tuned model was developed against a separately generated corpus version, which Section 3.5.3 treats as a disclosed limitation rather than a controlled condition. Metrics, parameters and confusion matrices are logged to MLflow, which keeps the comparison auditable and lets individual runs be recovered later.

Objective 4 is met by running BERTopic on the negative reviews of each platform separately, then merging topics across platforms by clustering their centroid vectors and naming each merged group with a language model. Reviews are assigned to categories in two steps: an embedding comparison shortlists a few likely categories, then a zero-shot entailment model ranks that shortlist. A confidence threshold leaves weak matches unclassified instead of forcing a label, which keeps a known unknown visible in the results — and, as Chapter 4 shows, that unknown turns out to be large enough to be a finding in its own right.

Objective 5 is met by indexing the corpus with a sentence-transformer model in FAISS, retrieving a small number of reviews per question, and passing them to the language model under instructions to answer only from that text. Each retrieved review is tagged with its platform, rating and sentiment before it reaches the model, and review identifiers are returned with every answer. The same questions are then put to the language model with no retrieval at all, which provides the baseline the comparison in RQ2 depends on.

Objective 6 uses three kinds of evaluation, chosen by what each stage allows. Sentiment classification has labels, so standard classification metrics apply, supported by McNemar's exact test for pairwise comparison. The taxonomy has none, so quality is argued from how tightly reviews cluster around their category, how evenly categories are populated, how they spread across platforms, and how many reviews fall below the confidence threshold. The question-answering component is scored with RAGAS. Two limitations should be recorded at the outset: the evaluation covers five questions, and the model used to judge answers is the same one that produces them, which introduces a self-assessment bias. Both were accepted because of free-tier API limits, and both are reported openly in Chapter 5 rather than being allowed to inflate the apparent strength of the generation results.

The tools are Python with pandas and scikit-learn for data handling and classical models, Hugging Face Transformers for RoBERTa and DistilBERT, VADER for the lexicon baseline, BERTopic with UMAP and HDBSCAN for topic discovery, sentence-transformers for embeddings, FAISS for the vector index, LangChain for the retrieval chain, Groq's hosted Llama 3.1 model for generation, RAGAS for generation metrics, and MLflow for experiment tracking. Random seeds are fixed throughout, dependency versions are pinned, and unit tests cover the topic-merging and category-shortlisting logic.

On ethics, all three datasets are publicly available and were collected by their original publishers, so no new data was gathered and no human participants were involved. The reviews contain no special-category personal data, reviewer names are not used, and reviews are referenced by identifier only. Two risks are worth recording. Review text occasionally contains personal details written by the reviewer, so no attempt is made to link reviews to individuals. Second, generated summaries describe named businesses, and acting on an unsupported summary could affect one unfairly. Returning source reviews with every answer is a partial safeguard, since a claim can be checked before it is used.

---

## 1.4 Dissertation Outline

**Chapter 2 — Literature Review** examines work on sentiment classification, topic modelling, dense retrieval, retrieval-augmented generation and explainable AI. It compares approaches instead of listing them, and identifies the two gaps this study addresses: the shortage of controlled comparison at each pipeline stage on multi-domain data, and the shortage of verification of generated output.

**Chapter 3 — Research Approach** sets out the methodology in full. It describes the datasets and preprocessing, the experimental design that makes the comparison controlled, the models and hyperparameters used, and the reasoning behind each choice, including the alternatives that were rejected and why.

**Chapter 4 — Data Analysis** reports the analytical work and its results: exploratory analysis of the three sources, the sentiment classification experiments and their diagnostic follow-ups, topic discovery for each platform, construction and measurement of the merged taxonomy, and the retrieval and generation evaluations.

**Chapter 5 — Discussion** interprets those results. It answers each research question in turn, compares the findings with the literature reviewed in Chapter 2, and examines the three results that run against expectation: a pretrained transformer losing to a linear model, a taxonomy in which a catch-all category absorbs more than a quarter of sampled reviews, and a standalone language model scoring higher than the retrieval-augmented system on answer relevancy while having no evidential basis for its answers. The limitations of the evaluation design itself are discussed here rather than deferred.

**Chapter 6 — Conclusion** reviews the aim and each objective, states what the study established and what it did not, and sets out future work drawn from the limitations actually encountered: a manually labelled evaluation set for the categorisation stage, a substantially larger question set and an independent judge model for the generation metrics, retraining the fine-tuned classifier on the current corpus version to remove the disclosed split limitation, and comparison against an alternative zero-shot classifier.
