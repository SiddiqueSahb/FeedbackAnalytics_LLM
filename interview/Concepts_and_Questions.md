# FeedbackIQ — Concept Inventory & Interview Question Bank

Questions only, no answers. Work through them yourself; the ones you can't answer out loud in two minutes are your revision list.

---

# PART 1 — CONCEPTS YOUR PROJECT ACTUALLY USES

Everything below appears somewhere in your code or results. If an interviewer reads your README, any of it is fair game.

## Classical machine learning
TF-IDF vectorisation · n-grams (you use 1–2) · vocabulary capping · Naive Bayes · Logistic Regression · stratified train/test splitting · StratifiedKFold cross-validation · class imbalance · label encoding · fitting the vectoriser on train only (leakage prevention)

## Transformers and deep learning
Self-attention · encoder-only architectures · BERT · RoBERTa · DistilBERT · DeBERTa-v3 · knowledge distillation · pretraining vs fine-tuning · transfer learning · domain shift / distribution mismatch · tokenisation and subword vocabularies · max sequence length and truncation · learning rate scheduling · epochs, batch size, gradient accumulation · overfitting detection via validation loss · best-checkpoint selection · fp16 / mixed precision · Hugging Face Trainer API

## Evaluation and statistics
Accuracy vs macro F1 vs weighted F1 · precision/recall trade-off · per-class metrics · confusion matrices (raw and row-normalised) · held-out test sets · data leakage and test-set contamination · McNemar's exact test · paired vs unpaired tests · p-values and significance thresholds · why non-significance ≠ equivalence · error analysis · diagnostic experiments · ablation studies · sampling and stratification · bimodal distributions and why means mislead

## Unsupervised learning and topic modelling
LDA (as the baseline you argue against) · BERTopic · sentence embeddings for clustering · UMAP dimensionality reduction · HDBSCAN density-based clustering · noise/outlier assignment · c-TF-IDF topic representation · minimum topic size · topic coherence · topic separation · cluster merging and consolidation · evaluation without ground truth

## Zero-shot classification
Natural language inference (NLI) · entailment as classification · hypothesis construction and label verbalisation · zero-shot vs few-shot vs fine-tuned · embedding-based candidate shortlisting · confidence thresholds and abstention

## Vector search and retrieval
Dense vs sparse retrieval · sentence-transformers · embedding dimensionality (384) · vector normalisation · cosine similarity vs L2 vs inner product · the ‖a−b‖² = 2 − 2·a·b identity · FAISS index types (Flat, IVF, IVF-PQ, HNSW) · exact vs approximate nearest neighbour · recall/latency trade-off · product quantisation · Maximal Marginal Relevance (MMR) · diversity vs relevance · metadata filtering · precision@k, recall@k, MRR, nDCG · bi-encoders vs cross-encoders · reranking

## RAG and LLMs
Retrieval-augmented generation · grounding · hallucination · context window · chunking strategies · prompt engineering · system vs user prompts · temperature · few-shot prompting · query reformulation for multi-turn · conversational memory · guardrails and scope classification · abstention/refusal design · RAGAS (faithfulness, answer relevancy, context precision, context recall) · LLM-as-judge and its biases · prompt injection · semantic caching

## Backend and systems engineering
REST API design · FastAPI · async/await and the event loop · blocking calls and thread offloading · Pydantic validation · OpenAPI · middleware · exception handling · structured logging · API key authentication · constant-time comparison · CORS · rate limiting · retry with backoff and jitter · idempotency · health checks (liveness vs readiness) · LRU caching · thread safety and locks · lazy loading

## MLOps and deployment
MLflow experiment tracking · parameters, metrics, artefacts, tags · model registry concepts · reproducibility (seeds, pinned dependencies) · data and model versioning · Docker · multi-stage builds · docker-compose · volumes vs baked-in artefacts · environment configuration and secrets management · CI/CD · GitHub Actions · linting · unit and integration testing · monitoring and drift detection · blue-green deployment · horizontal scaling

## Data engineering
Parquet vs CSV · schema unification across heterogeneous sources · text cleaning and normalisation · deduplication · handling missing values · label derivation from proxies (star ratings) · sampling strategies · memory-efficient processing

---

# PART 2 — QUESTION BANK

## Round 1 — Project walkthrough

1. Walk me through your project in two minutes.
2. What problem does it solve, and for whom?
3. Why three separate stages instead of one end-to-end model?
4. What was the hardest technical decision you made?
5. What's the most interesting thing you learned?
6. What would you do differently if you started again?
7. What's the weakest part of your project?
8. How would you know if this system was working in production?
9. Who would use this, and what would they do with the output?
10. How long did this take, and where did the time actually go?
11. What did you build yourself versus take off the shelf?
12. If I gave you two more months, what would you build?

## Round 2 — Data and preprocessing

13. Where did your data come from and why those three sources?
14. Your labels come from star ratings. What's wrong with that?
15. How does "neutral" differ across Amazon, Yelp and Twitter?
16. Walk me through your preprocessing pipeline.
17. Why did you filter reviews under five characters?
18. How did you unify three different schemas?
19. Your corpus is 70% Yelp. What problems does that cause?
20. How did you split the data, and why stratified?
21. What is data leakage? Where could it occur in your pipeline?
22. Your test set is contaminated for one model. Explain exactly how.
23. Why can't you quantify the contamination extent?
24. How would you prevent that in a production ML system?
25. Why Parquet rather than CSV?

## Round 3 — Classical ML

26. Explain TF-IDF. Why not just word counts?
27. Why did you use 1–2 n-grams rather than unigrams only?
28. Why cap the vocabulary at 50,000 features?
29. Why fit the vectoriser on training data only?
30. Explain Naive Bayes. What's "naive" about it?
31. Why does logistic regression beat Naive Bayes on your data?
32. Your logistic regression is within two points of a fine-tuned transformer. Justify the transformer.
33. When would you choose a classical model over a transformer in production?
34. What's the inference cost difference between the two?
35. How would you tune the regularisation on your logistic regression?

## Round 4 — Transformers and fine-tuning

36. Explain self-attention. Why does it help for sentiment?
37. What's the computational complexity of attention with respect to sequence length?
38. Explain the difference between BERT, RoBERTa and DistilBERT.
39. What is knowledge distillation and what does it cost you?
40. Why DistilBERT rather than DeBERTa-v3 or RoBERTa for fine-tuning?
41. You used DeBERTa-v3 elsewhere in the project. Why the inconsistency?
42. Walk me through your fine-tuning configuration and justify two choices.
43. Your validation loss rose after epoch 2 while training loss kept falling. What's happening?
44. Why did you train four epochs if epoch 2 was best?
45. What is best-checkpoint selection and why does it matter?
46. Explain what fine-tuning actually changes inside the model.
47. What is catastrophic forgetting? Is it a risk here?
48. Your pretrained RoBERTa scored below logistic regression. Diagnose it.
49. Why does RoBERTa beat everything on tweets and lose badly on Yelp?
50. What is domain shift and how would you detect it before deployment?
51. How would you adapt a model to a new domain without full retraining?
52. What's the memory footprint of your fine-tuned model at inference?
53. How would you reduce inference latency by half?
54. What is quantisation? What would you expect it to cost you here?

## Round 5 — Evaluation and statistics

55. Explain macro F1 versus weighted F1. Why is macro your headline?
56. When is accuracy a misleading metric?
57. Walk me through a confusion matrix from your results.
58. Why row-normalise a confusion matrix?
59. What is McNemar's test and why use it rather than comparing F1 scores?
60. Why a paired test rather than an unpaired one?
61. Two of your models weren't significantly different. What does that mean — and what does it *not* mean?
62. What's a p-value, in plain terms?
63. You report a mean of 0.400 across five values that are 1, 1, 0, 0, 0. What's wrong with that?
64. How many evaluation questions would you need for a meaningful confidence interval?
65. Describe your binary re-evaluation experiment and what it proved.
66. How did you evaluate the topic model with no ground truth?
67. Why is topic coherence a contested metric?
68. What is an ablation study and why does it matter?
69. Design an ablation to test whether your reranker helps.
70. Your ablation improved one metric and hurt another. How do you decide whether to ship it?

## Round 6 — Topic modelling and clustering

71. Explain BERTopic's pipeline end to end.
72. Why BERTopic rather than LDA?
73. What does UMAP do, and why reduce dimensions before clustering?
74. Why HDBSCAN rather than k-means?
75. What does it mean that HDBSCAN labels some points as noise?
76. Why run BERTopic per platform rather than on the pooled corpus?
77. Amazon gave 69 topics from 58k documents; Yelp gave 18 from 150k. Explain.
78. How did you merge 121 topics into 24 categories?
79. Why 24? How would you choose that number properly?
80. 27.6% of your held-out sample is unclassified. Is that good or bad?
81. Your categoriser assigns positive reviews to complaint categories. Why?
82. Explain zero-shot classification via natural language inference.
83. Why shortlist with embeddings before applying entailment?
84. Why pass the category description rather than the category name as the hypothesis?

## Round 7 — Vector search and retrieval

85. Explain how a semantic search engine works, end to end.
86. Why FAISS rather than Pinecone, Weaviate, Chroma or pgvector?
87. You use IndexFlatL2 with normalised vectors. Why L2 rather than inner product?
88. Derive the relationship between L2 distance and cosine similarity for unit vectors.
89. What's the time complexity of a flat index search? When does it break?
90. Explain IVF, HNSW and product quantisation. When would you use each?
91. What's the trade-off you accept when moving from exact to approximate search?
92. Why all-MiniLM-L6-v2? Why not a larger embedding model?
93. What does the embedding dimension of 384 buy and cost you?
94. Explain your chunking strategy and its alternatives.
95. What is MMR and why use it instead of plain top-k similarity?
96. What does lambda_mult control in MMR?
97. Explain your two-pass grounded retriever and why MMR alone wasn't enough.
98. Explain precision@k, recall@k, MRR and nDCG.
99. Why is nDCG position-aware and why does that matter?
100. Your retrieval is fast with precision@3 of 0.90, but context precision is 0.416. Reconcile those.
101. Bi-encoder versus cross-encoder — what's the difference and where does each belong?
102. Where would you add a reranker and what would it cost?
103. How would you build a labelled retrieval evaluation set without annotating 642,692 reviews?
104. How do you keep the index fresh when new reviews arrive?

## Round 8 — RAG and LLMs

105. What is RAG and what problem does it solve?
106. What are the failure modes of a RAG system?
107. Walk me through what happens when a user asks your chatbot a question.
108. Explain your two scope guards and why the order matters.
109. Your keyword guard misses most out-of-scope questions. Is that a problem?
110. What happens if the LLM API is down — what protection remains?
111. Explain your similarity threshold. How was 0.35 chosen?
112. How would you calibrate that threshold properly?
113. What's the cost of setting it too high? Too low?
114. Explain RAGAS faithfulness and answer relevancy.
115. Explain context precision and context recall.
116. Why does context recall need a reference answer?
117. Your RAG scored lower on relevancy than an ungrounded baseline. Why keep RAG?
118. What are the weaknesses of LLM-as-judge evaluation?
119. The same model generates and grades your answers. What's wrong with that?
120. How would you validate that your judge is reliable?
121. What is prompt injection and how does it apply to your system?
122. Your retrieved reviews are user-generated content going into a prompt. What's the risk?
123. Why temperature 0.2 rather than 0 or 0.7?
124. Explain query reformulation for multi-turn conversations. What does it cost?
125. How would you implement semantic caching, and what would it save?
126. How would you stream responses, and why would you want to?
127. When should a RAG system refuse to answer?
128. How do you distinguish "correctly refused" from "wrongly refused" in your metrics?

## Round 9 — Coding

129. Implement cosine similarity and top-k retrieval from scratch. No FAISS, no sklearn.
130. Why would you use `np.argpartition` instead of `np.argsort` here?
131. Implement MMR selection.
132. Implement macro F1 from a list of predictions and labels, without sklearn.
133. Implement McNemar's test from a contingency table.
134. Implement an LRU cache. What's the eviction cost?
135. Write an async FastAPI endpoint that doesn't block the event loop.
136. Implement retry with exponential backoff and jitter. Retry only what should be retried.
137. Given a list of documents and a query, write a function that returns only those above a similarity threshold, deduplicated by ID.
138. Write a function that batches a large list for processing with a bounded memory footprint.
139. Design your retrieval pipeline as a class hierarchy. Show me the interfaces.
140. Write a test that would have caught your split contamination.

## Round 10 — System design

141. Scale this to 100 million reviews, 500 QPS, p99 under 300ms.
142. Your index is 990MB and loaded per process. Fix it.
143. How do you add a review without rebuilding the index?
144. Design multi-tenancy — separate customers, isolated data.
145. Design monitoring for this system. What pages someone at 3am?
146. How would you A/B test your metadata-tagging change in production?
147. Design a retraining pipeline that triggers when performance degrades.
148. How do you detect model drift with no ground-truth labels in production?
149. Design the system so a bad model version can be rolled back in one minute.
150. Where would you put a cache, and what would you cache?
151. How would you reduce LLM cost by 50%?
152. Design an evaluation harness that runs in CI on every prompt change.

## Round 11 — Engineering and production

153. Walk me through your Docker setup. Why two containers?
154. Why mount data and models as volumes rather than baking them in?
155. How does that design fail on a serverless platform?
156. Explain the difference between a liveness and a readiness check.
157. Your health endpoint checks file existence. What's wrong with that?
158. Why `asyncio.to_thread` around FAISS search?
159. What happens if you write `async def` around synchronous blocking work?
160. Explain `lru_cache` plus a threading lock. Why both?
161. Why pin uvicorn to one worker in your container?
162. Security-review your API. What's wrong with it?
163. Why constant-time comparison for an API key?
164. What's in your CI pipeline, and what should be?
165. Your `tests/` folder has almost no assertions. Defend or concede.
166. What does MLflow track in your project, and what doesn't it track?
167. How would you version your data and model artefacts?
168. What is a model registry and why would you want one?
169. How would you structure logs so you can debug a single user's failed request?
170. Your logs record that a refusal happened but not why. Why does that matter?

## Round 12 — Attacks on your work

171. Your test set is contaminated. Why should I trust any of your numbers?
172. You shipped a magic number. 0.35 isn't engineering, it's guessing.
173. Five evaluation questions is not an evaluation.
174. Your system takes 1.5 seconds and costs an API call per query. Nobody will use it.
175. Flat FAISS with a single-process load. This doesn't scale past one instance.
176. Your scope guard rejected a valid question and you only noticed by accident. What else is broken that you haven't noticed?
177. You wrote your own reference answers and then measured retrieval against them. Isn't that circular?
178. Your project has no novel contribution. Why should I be impressed?
179. A logistic regression gets within two points. Why did you waste three hours on a GPU?
180. You didn't run three of your four planned ablations. Why should I believe the one you did run?

## Round 13 — Behavioural

181. Tell me about a bug that took you a long time to find.
182. Tell me about a time you were wrong about something technical.
183. Tell me about a result that surprised you.
184. How do you decide when something is good enough to ship?
185. Tell me about a trade-off you made under time pressure.
186. How do you approach a problem you don't know how to solve?
187. Tell me about feedback you received that changed how you work.
188. How would you explain your project to a non-technical stakeholder?
189. What would you tell a junior engineer starting a similar project?
190. Why do you want to be an ML engineer rather than a data scientist?

---

# PART 3 — HOW TO USE THIS

Answer out loud, timed, without notes. Two minutes each for conceptual questions, fifteen for coding.

Mark each one:
- **Green** — could answer in an interview tomorrow
- **Amber** — know the idea, would fumble the delivery
- **Red** — don't actually know this

Work the reds first, then rehearse the ambers until they're automatic. The greens don't need practice.

If you can't answer a question about your own project, that's not a revision gap — that's something to go and check in the code before an interviewer finds it.
