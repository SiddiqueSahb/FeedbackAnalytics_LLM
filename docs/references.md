# References (APA 7)

Entries in Parts A and B were verified against the publisher record, ACL Anthology, arXiv or
DOI resolver. Nothing in those sections is reconstructed from memory.

Part C lists claims still lacking a located source. Those must either be sourced or the
associated sentence removed before submission — they are not to be filled in by guesswork.

---

## Part A — Method and technology sources (cited in Chapters 1–4)

**Sentiment analysis**

Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis
of social media text. *Proceedings of the International AAAI Conference on Web and Social Media,
8*(1), 216–225. https://doi.org/10.1609/icwsm.v8i1.14550

Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. *Foundations and Trends in
Information Retrieval, 2*(1–2), 1–135. https://doi.org/10.1561/1500000011

**Transformer models**

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep
bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of
the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies, Volume 1 (Long and Short Papers)* (pp. 4171–4186). Association for Computational
Linguistics. https://doi.org/10.18653/v1/N19-1423

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., &
Stoyanov, V. (2019). *RoBERTa: A robustly optimized BERT pretraining approach* (arXiv:1907.11692).
arXiv. https://arxiv.org/abs/1907.11692

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT:
Smaller, faster, cheaper and lighter* (arXiv:1910.01108). arXiv. https://arxiv.org/abs/1910.01108

Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified
benchmark and comparative evaluation for tweet classification. In *Findings of the Association
for Computational Linguistics: EMNLP 2020* (pp. 1644–1650). Association for Computational
Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.148
> Cite for `cardiffnlp/twitter-roberta-base-sentiment-latest`. Documents the Twitter pretraining
> data, which is the evidence behind the domain-mismatch argument for RQ1.

**Topic modelling**

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine
Learning Research, 3*, 993–1022. https://jmlr.org/papers/volume3/blei03a/blei03a.pdf

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*
(arXiv:2203.05794). arXiv. https://arxiv.org/abs/2203.05794

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013),
Lecture Notes in Computer Science, Vol. 7819* (pp. 160–172). Springer.
https://doi.org/10.1007/978-3-642-37456-2_14
> HDBSCAN, the clustering step inside BERTopic.

McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform manifold approximation and
projection for dimension reduction* (arXiv:1802.03426). arXiv. https://arxiv.org/abs/1802.03426
> The dimensionality reduction step inside BERTopic.

**Embeddings, retrieval and zero-shot classification**

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics.
https://doi.org/10.18653/v1/D19-1410
> The sentence-transformer family behind `all-MiniLM-L6-v2`, used for the FAISS index and
> BERTopic's document embeddings.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE
Transactions on Big Data, 7*(3), 535–547. https://doi.org/10.1109/TBDATA.2019.2921572
> FAISS, the vector index used for retrieval.

Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2023). *Building efficient universal
classifiers with natural language inference* (arXiv:2312.17543). arXiv.
https://arxiv.org/abs/2312.17543
> The zero-shot NLI classifier family behind `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
> used for assigning reviews to complaint categories.

**Retrieval-augmented generation**

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for
knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33*
(pp. 9459–9474). Curran Associates.
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., &
Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys,
55*(12), Article 248, 1–38. https://doi.org/10.1145/3571730

Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of
retrieval augmented generation. In *Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association
for Computational Linguistics. https://doi.org/10.18653/v1/2024.eacl-demo.16

---

## Part B — Sources located during the reference audit (Chapter 2 and 5 citations)

**Topic model evaluation and its critique**

Hoyle, A. M., Goel, P., Hian-Cheong, A., Peskov, D., Boyd-Graber, J., & Resnik, P. (2021). Is
automated topic model evaluation broken? The incoherence of coherence. In *Advances in Neural
Information Processing Systems 34* (pp. 2018–2033). Curran Associates.
https://arxiv.org/abs/2107.02173
> Resolves **[V17]**. The central critique that automated coherence does not reliably reproduce
> human judgement, and that coherence developed for classical models was never validated on
> neural models.

Doogan, C., & Buntine, W. (2021). Topic model or topic twaddle? Re-evaluating semantic
interpretability measures. In *Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies* (pp. 3824–3848).
Association for Computational Linguistics. https://aclanthology.org/2021.naacl-main.300/
> Resolves **[V18]**. Note the year: this is a **2021 NAACL** paper, not a 2024 one.

Lim, J. P., & Lauw, H. W. (2024). Aligning human and computational coherence evaluations.
*Computational Linguistics, 50*(3), 893–952. https://doi.org/10.1162/coli_a_00518
> Resolves **[V19]**, the *Computational Linguistics* 2024 paper on the correspondence between
> automated coherence metrics and human judgement. **See the correction note in Part D — the
> draft attributed this paper to the wrong authors.**

Rahimi, H., Hoover, J. L., Mimno, D., Naacke, H., Constantin, C., & Amann, B. (2024).
Contextualized topic coherence metrics. In *Findings of the Association for Computational
Linguistics: EACL 2024* (pp. 1889–1900). Association for Computational Linguistics.
https://aclanthology.org/2024.findings-eacl.123/
> Resolves **[V20]**. Embedding-based coherence in place of co-occurrence counts.

**Topic modelling comparisons**

Turan, S. C., Yildiz, K., & Büyüktanir, B. (2024). Comparison of LDA, NMF and BERTopic topic
modeling techniques on Amazon product review dataset: A case study. In *Computing, Internet of
Things and Data Analytics (ICCIDA 2023), Studies in Computational Intelligence*. Springer.
https://doi.org/10.1007/978-3-031-53717-2_3
> Resolves **[V9]**. Reports all three algorithms usable with BERTopic producing the more
> meaningful results.

Egger, R., & Yu, J. (2022). A topic modeling comparison between LDA, NMF, Top2Vec, and BERTopic
to demystify Twitter posts. *Frontiers in Sociology, 7*, Article 886498.
https://doi.org/10.3389/fsoc.2022.886498
> Additional short-text comparison supporting the document-length argument in Section 2.1.2.
> Not previously cited; recommended addition, since it directly evidences BERTopic's advantage
> on short informal text.

Mishra, M. (2026). Comparative analysis of BERTopic versus LDA: Topic modelling in marketing
research. *Journal of Creative Communications*. Advance online publication.
https://doi.org/10.1177/14413582251399667
> Resolves **[V10]**. Published online ahead of issue; APA advance-online form used. **Verify
> the journal title and whether an issue has since been assigned before submitting** — the
> record was located but the container title should be confirmed against the publisher page.

**Sentiment classification comparisons**

Areshey, A., & Mathkour, H. (2023). Transfer learning for sentiment classification using
bidirectional encoder representations from transformers (BERT) model. *Sensors, 23*(11), 5232.
https://doi.org/10.3390/s23115232
> This is the source of the Yelp binary-polarity benchmark figures quoted in Chapter 5,
> Table 5.1. **Note the year is 2023, not 2024** — correct the in-text citation.

Areshey, A., & Mathkour, H. (2024). Exploring transformer models for sentiment classification:
A comparison of BERT, RoBERTa, ALBERT, DistilBERT, and XLNet. *Expert Systems*.
https://doi.org/10.1111/exsy.13701
> A second, later paper by the same authors. Useful additional breadth for Section 2.1.2;
> check which of the two you actually intend to cite in each place.

**RAG evaluation and LLM-as-judge**

Chen, J., Lin, H., Han, X., & Sun, L. (2024). Benchmarking large language models in
retrieval-augmented generation. *Proceedings of the AAAI Conference on Artificial Intelligence,
38*(16), 17754–17762. https://doi.org/10.1609/aaai.v38i16.29728
> Resolves **[V14]**. Establishes LLM sensitivity to noise and irrelevant retrieved passages,
> and evaluates noise robustness, negative rejection, information integration and counterfactual
> robustness — directly relevant to the context-precision finding in Section 5.3.3.

Yu, H., Gan, A., Zhang, K., Tong, S., Liu, Q., & Liu, Z. (2024). *Evaluation of
retrieval-augmented generation: A survey* (arXiv:2405.07437). arXiv.
https://arxiv.org/abs/2405.07437
> Resolves **[V13]**. The source for the argument that aggregate RAG scores cannot diagnose
> whether a failure originated in retrieval or generation.

Gu, J., Jiang, X., Shi, Z., Tan, H., Zhai, X., Xu, C., Li, W., Shen, Y., Ma, S., Liu, H.,
Wang, Y., & Guo, J. (2024). *A survey on LLM-as-a-judge* (arXiv:2411.15594). arXiv.
https://arxiv.org/abs/2411.15594
> Resolves **[V15]**. Load-bearing: supports the self-enhancement bias limitation stated in
> Sections 2.2.3, 3.6.5 and 5.3.4.

Li, H., Dong, Q., Chen, J., Su, H., Zhou, Y., Ai, Q., Ye, Z., & Liu, Y. (2024). *LLMs-as-judges:
A comprehensive survey on LLM-based evaluation methods* (arXiv:2412.05579). arXiv.
https://arxiv.org/abs/2412.05579
> Resolves **[V16]**. Load-bearing, as above.

---

## Part C — Claims still without a located source

Each row is a claim currently made in the text with no verifiable citation behind it. **Do not
invent an entry for any of these.** Either locate the source, or delete/soften the sentence so
that no unsupported claim remains.

| ID | Claim as currently written | Where | Recommended action |
|---|---|---|---|
| V1 | Amazon sentiment comparison across ML, DL and transformers, cited as "*Electronics* (2024)" | §2.1.2, Table 2.1, Table 5.1 | Journal title only. Either locate it or replace with Areshey & Mathkour (2023/2024), which is verified and makes the same point |
| V2 | ML vs pre-trained language models on product reviews, "*Big Data and Cognitive Computing* (2024)" | §2.1.2 | Journal title only. Replaceable by the same substitution as V1 |
| V3 | Amazon Magazine Subscriptions 2023 dataset, several embedding schemes | §2.1.2 | No citation at all. Delete the sentence unless sourced |
| V4 | Zhang et al. (2023), feature projection with multi-source attention | §2.1.2, Table 2.1 | Cited in text, absent from list. Locate or replace the cross-domain example |
| V5 | Adversarial gradient-reversal cross-domain work on Amazon/Yelp/IMDB | §2.1.2 | No citation. Delete or source |
| V6 | Retail ABSA corpus, cited as arXiv:2508.17994 | §2.1.2 | Bare identifier. Resolve on arXiv and complete, or delete |
| V7 | Entailment vs similarity for unsupervised classification, arXiv:2211.16285 | §2.1.2 | Bare identifier. Resolve and complete |
| V8 | Zero-shot benchmarking across cross-encoders, embeddings, rerankers, generative models | §2.1.2 | No citation. Delete or source |
| V11 | BERTopic on online communities, arXiv:2412.14486 | §2.1.2 | Bare identifier. Resolve and complete |
| **V12** | **Indonesian customer complaint study: LDA coherence 0.68 vs BERTopic 0.51** | §2.1.2, Table 2.1, Table 5.1 | **Load-bearing — these two figures are quoted in Chapter 5. If the source cannot be located, both numbers and the comparison built on them must be removed from Table 5.1 and §5.4.2** |
| V-fin | Financial sentiment study (2023): BERT 87% accuracy, macro F1 0.85, three-class | Table 5.1, §5.4.2 | **Load-bearing** — this is the three-class benchmark your DistilBERT result is compared against. Source it or remove that row and the paragraph interpreting it |
| V-16to56 | "False discovery rates between 16% and 56%" | §2.1.3, §5.2.4 | **Verify this figure against Lim & Lauw (2024) before retaining it.** The paper is now located, but this specific range was not confirmed during the audit. If it cannot be confirmed in the text, state the finding qualitatively instead — the argument does not depend on the exact numbers |

---

## Part D — Corrections required in the chapter text

These are errors in the drafts, not missing references. Each needs a find-and-replace.

1. **Wrong attribution, Chapter 2 §2.1.3 and Chapter 5 §5.2.4.** The *Computational Linguistics*
   2024 coherence paper is by **Lim and Lauw**, not "Doogan-Poet Sirakoff & Buntine". Replace
   `(Doogan-Poet Sirakoff & Buntine, 2024)` with `(Lim & Lauw, 2024)` in both places.

2. **Wrong year, Chapter 2 §2.1.3 and Table 2.1.** Doogan and Buntine's critique is **2021**
   (NAACL), not 2024. If you cite it, cite it as `(Doogan & Buntine, 2021)`.

3. **Wrong year, Chapter 5 Table 5.1.** The Yelp binary benchmark row cites
   "Areshey & Mathkour (2024)". The *Sensors* paper carrying those figures is **2023**. Confirm
   which of the two papers your quoted percentages came from and cite that one.

4. **Chapter 2 §2.5** currently lists V1–V20 as outstanding. Update it: V9, V10, V13, V14, V15,
   V16, V17, V18, V19 and V20 are now resolved in Part B above. Only the rows in Part C remain.

---

## Part E — Coverage check against the marking scheme

The Chapters 1 & 2 criterion (20%) requires, at Distinction, "an excellent review of relevant
material, including research papers and advanced materials" with "complete and correct"
references. Current position:

- **Method sources: complete.** Every model, library and metric used in the project has a
  verified primary citation.
- **Critical-analysis sources: complete.** The topic-model evaluation critique (Hoyle; Doogan &
  Buntine; Lim & Lauw; Rahimi et al.), the LLM-as-judge bias literature (Gu et al.; Li et al.)
  and the RAG evaluation critique (Yu et al.; Chen et al.) are all now sourced. These are the
  entries that support the chapter's strongest analytical passages.
- **Applied comparison sources: incomplete.** V1–V8 and V11 remain unsourced. These are the
  breadth citations in §2.1.2. The chapter's argument survives without them, but the "range of
  research papers" expectation is better met with them present.
- **Benchmark sources: two load-bearing gaps.** V12 and V-fin both supply numbers reused in
  Chapter 5. Unlike the breadth citations, these cannot simply be dropped without also removing
  the comparisons built on them.

**Priority order:** V-fin and V12 first, because Chapter 5 quotes their figures. Then V4 and V1,
which appear in Table 2.1. The remainder are breadth citations and can be dropped with minor
edits if they cannot be located.
