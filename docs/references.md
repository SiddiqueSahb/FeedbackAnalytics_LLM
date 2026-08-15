# References (APA 7)

Every entry below was verified against the publisher or ACL Anthology record. Nothing here is
reconstructed from memory. Entries are grouped by where they are cited so that placeholders in
the chapter drafts can be replaced directly.

---

## Cited in Chapter 1

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

**Topic modelling**

Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. *Journal of Machine
Learning Research, 3*, 993–1022. https://jmlr.org/papers/volume3/blei03a/blei03a.pdf

Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a class-based TF-IDF procedure*
(arXiv:2203.05794). arXiv. https://arxiv.org/abs/2203.05794

**Retrieval-augmented generation and its evaluation**

Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of
retrieval augmented generation. In *Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association
for Computational Linguistics. https://doi.org/10.18653/v1/2024.eacl-demo.16

Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y. J., Madotto, A., &
Fung, P. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys,
55*(12), Article 248, 1–38. https://doi.org/10.1145/3571730

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for
knowledge-intensive NLP tasks. In *Advances in Neural Information Processing Systems 33*
(pp. 9459–9474). Curran Associates. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html

---

## Additional references for Chapters 3 and 4 (methods actually used in the code)

Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., & Neves, L. (2020). TweetEval: Unified
benchmark and comparative evaluation for tweet classification. In *Findings of the Association
for Computational Linguistics: EMNLP 2020* (pp. 1644–1650). Association for Computational
Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.148
> Cite this for `cardiffnlp/twitter-roberta-base-sentiment-latest`. It documents the Twitter
> pretraining data, which is the evidence behind the domain-mismatch argument for RQ1.

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on
hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013),
Lecture Notes in Computer Science, Vol. 7819* (pp. 160–172). Springer.
https://doi.org/10.1007/978-3-642-37456-2_14
> HDBSCAN, the clustering step inside BERTopic.

Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE
Transactions on Big Data, 7*(3), 535–547. https://doi.org/10.1109/TBDATA.2019.2921572
> FAISS, the vector index used for retrieval.

Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2023). *Building efficient universal
classifiers with natural language inference* (arXiv:2312.17543). arXiv.
https://arxiv.org/abs/2312.17543
> The zero-shot NLI classifier family behind `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`,
> used for assigning reviews to complaint categories.

McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform manifold approximation and
projection for dimension reduction* (arXiv:1802.03426). arXiv. https://arxiv.org/abs/1802.03426
> The dimensionality reduction step inside BERTopic.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese
BERT-networks. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP)* (pp. 3982–3992). Association for Computational Linguistics.
https://doi.org/10.18653/v1/D19-1410
> The sentence-transformer family behind `all-MiniLM-L6-v2`, used for both the FAISS index and
> BERTopic's document embeddings.

---

## Placeholder replacement map for Chapter 1

| Placeholder in draft | Replace with |
|---|---|
| (Hutto & Gilbert, 2014) | unchanged — now a real entry |
| (Pang & Lee, 2008) | unchanged |
| (Devlin et al., 2019; Sanh et al., 2019; Liu et al., 2019) | unchanged |
| (Blei et al., 2003) | unchanged |
| (Grootendorst, 2022) | unchanged |
| (Ji et al., 2023) | unchanged |
| (Lewis et al., 2020) | unchanged |
| (Es et al., 2024) | unchanged |

All eight in-text citations in Chapter 1 already use the correct author-year form, so no edits to
the chapter body are required. Only this reference list needed to exist.

---

## Still to source for Chapter 2

Chapter 2 needs breadth beyond the method papers above, since the marking scheme rewards a
"range of research papers". The following areas currently have no citation and will need
searching:

- Applied studies of customer feedback or complaint analysis at scale, ideally cross-platform.
- Empirical comparisons of classical machine learning against transformers on review data,
  including any reporting that fine-tuning did not help.
- Work on topic model evaluation without ground truth (coherence measures such as C_v and NPMI,
  topic diversity).
- Studies evaluating RAG against non-retrieval baselines, and critiques of LLM-as-judge
  evaluation, which is directly relevant to the self-grading limitation in this project.
- Explainability and traceability in applied NLP systems.
