**School of Computer Science and Electronic Engineering**

**MSc Data Science**

**Academic Year 2024-2025**

***Intelligent Customer Feedback Analytics using Large Language Models: A Stage-Wise Empirical Comparison of Classical Machine Learning, Transformer-Based Models and Retrieval-Augmented Generation***

*A project report submitted by: Mohammad Asim Siddique [INSERT STUDENT ID]*

*A project supervised by: [INSERT SUPERVISOR NAME]*

A report submitted in partial fulfilment of the requirement for the degree of Master of Science

University of Surrey

School of Computer Science and Electronic Engineering

Guildford, Surrey GU2 7XH

United Kingdom.

Tel: +44 (0)1483 300800

\newpage

# ABSTRACT

Organisations collect far more customer feedback than they can read, yet methods such as lexicon scoring, classical machine learning, transformer models and retrieval-augmented generation are often selected on reputation rather than evidence. Existing comparative studies are commonly based on single datasets and aggregate metrics, which can conceal how methods behave across different types of feedback and where they fail.

This dissertation compares these approaches stage by stage using a corpus of 642,692 reviews spanning Amazon products, Yelp businesses and airline complaint tweets, within a common comparative framework. Five sentiment classifiers were evaluated using per-class, per-platform and confusion-matrix analysis alongside McNemar significance testing. BERTopic was used to discover complaint topics separately by platform before consolidation into a 24-category cross-domain taxonomy, assessed without ground truth through coherence, coverage and distribution measures. A retrieval-augmented question-answering pipeline over a FAISS index was compared with a standalone language model using four RAGAS metrics, and one design decision within that pipeline was ablated directly.

The central finding is that method ranking depends on domain. Fine-tuned DistilBERT achieved the highest clean held-out macro F1 of 0.8375, while pretrained RoBERTa scored 0.7853 on airline tweets but only 0.5544 on Yelp, outperforming both classical models on the platform closest to its training domain while underperforming logistic regression on the pooled evaluation. Diagnostic experiments showed this behaviour was concentrated in the neutral class and was consistent with a mismatch between label definitions rather than general model weakness. Fine-tuning's clearest benefit was cross-platform stability rather than peak accuracy. The taxonomy produced well-separated categories but left 27.6% of reviews unclassified and assigned some positive reviews to complaint categories. Retrieval augmentation achieved 0.540 faithfulness, a property not defined for the ungrounded baseline, while scoring lower on mean answer relevancy (0.610 against 0.753). Component-level metrics located retrieval, rather than generation, as the pipeline's main measured weakness. The executed ablation found metadata tagging to be a trade-off rather than a clean improvement: faithfulness rose by 0.118 while answer relevancy fell by 0.202, alongside one directly observed hallucinated platform citation that tagging prevented.

The study contributes a stage-wise cross-domain comparison, diagnostic analysis of anomalous results, an executed ablation of one retrieval design decision, and evidence showing how aggregate metrics conceal important failure modes. The generation evaluation was limited to five questions judged by the generating model, and three of four designed ablations were not carried out, so conclusions at that stage remain directional rather than definitive.

\newpage

# HIGHLIGHTS

- Sentiment method ranking reverses by platform: RoBERTa 0.7853 on tweets, 0.5544 on Yelp
- Pretrained transformer's failure isolated to one class; binary re-evaluation restores ranking
- Fine-tuning buys cross-platform stability, not peak accuracy, across three feedback registers
- RAG scored lower on relevancy than an ungrounded baseline while being the only verifiable one
- Metadata-tagging ablation: faithfulness +0.118, answer relevancy −0.202 — a measured trade-off
- Multi-indicator evaluation exposed taxonomy defects a single coherence score would have hidden

\newpage

# ACKNOWLEDGEMENTS

[INSERT YOUR ACKNOWLEDGEMENTS HERE — your supervisor, family, friends, and anyone who supported the work.]

\newpage

Sign in the box below to certify that the work carried out is your own. By signing this box you are certifying that your dissertation is free from plagiarism. Make sure that you are fully aware of the Department guidelines on plagiarism (see the student handbook). The penalties if you are caught are severe. All material from other sources must be properly referenced and direct quotes must appear in quotation marks.

I certify that the work presented in the dissertation is my own unless referenced

Signature.................................................

Date...............................................

*Note to reader: the Table of Contents, List of Tables and List of Figures are Word fields. Select all (Ctrl+A) and press F9 to populate page numbers before printing.*

**TOTAL NUMBER OF WORDS: 25,611**

\newpage
