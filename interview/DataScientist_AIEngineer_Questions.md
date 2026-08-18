# Data Scientist & AI Engineer — Interview Question Bank

Questions only, no answers. Companion to `Concepts_and_Questions.md`, which covers the ML Engineer track and your project deep dive.

**Where your project helps.** For **AI Engineer** it's directly relevant — RAG, retrieval, evaluation, hallucination and guardrails are the daily work, and you have real numbers for all of them. For **Data Scientist** it helps less: DS loops lean on statistics, experimentation, SQL and product judgement, and your project touches those only through McNemar's test. Prepare accordingly.

---

# PART A — DATA SCIENTIST

## A1. Probability and statistics

1. Explain Bayes' theorem and give a real example of using it.
2. What's the difference between probability and likelihood?
3. Explain the Central Limit Theorem and why it matters in practice.
4. When would you use a t-test versus a chi-squared test versus a Mann-Whitney U test?
5. What is a p-value, precisely? What is it not?
6. What is statistical power and what drives it?
7. Explain Type I and Type II errors with a business example.
8. What is the multiple comparisons problem and how do you correct for it?
9. Explain the difference between MLE and MAP estimation.
10. What is a confidence interval, and what does 95% actually refer to?
11. Explain bootstrapping and when you'd reach for it.
12. What's the difference between correlation and covariance?
13. Explain Simpson's paradox with an example.
14. When is the median a better summary than the mean?
15. Your metric is bimodal. What summary statistics do you report?
16. Explain the bias-variance trade-off.
17. What distributions would you expect for: session length, revenue per user, number of purchases?
18. How do you detect and handle outliers? When should you not remove them?
19. Explain survivorship bias and give an example from a product context.
20. What is selection bias and how does it arise in observational data?
21. Explain regression to the mean and how it fools people.
22. How do you test whether two samples come from the same distribution?

## A2. Experimentation and A/B testing

23. Design an A/B test for a new recommendation algorithm.
24. How do you determine the sample size before running a test?
25. How long should you run an experiment, and what decides that?
26. What is a novelty effect and how do you detect it?
27. What is p-hacking and how do you prevent it structurally?
28. Explain peeking and why it inflates false positives.
29. What is a sequential test and when would you use one?
30. Your A/B test is significant but the effect size is tiny. What do you recommend?
31. Your treatment wins on the primary metric and loses on a guardrail metric. What now?
32. What are guardrail metrics and how do you choose them?
33. Explain network effects and interference in experiments. How do you handle them?
34. What is a switchback test and when is it appropriate?
35. How would you run an experiment when randomisation isn't possible?
36. Explain CUPED or another variance-reduction technique.
37. What is a holdout group and why keep one long-term?
38. How do you handle multiple simultaneous experiments on the same surface?
39. Your experiment shows no effect. Is that a failure? What do you report?
40. How would you A/B test a change that only affects 0.1% of users?

## A3. Causal inference

41. Why doesn't correlation imply causation? Give a concrete counterexample.
42. What is a confounder, and how do you control for one?
43. Explain difference-in-differences.
44. Explain propensity score matching and its assumptions.
45. What is an instrumental variable? Give an example.
46. Explain regression discontinuity design.
47. What is a natural experiment? Have you seen one?
48. When is observational analysis acceptable in place of an experiment?

## A4. SQL and data manipulation

49. Write a query to find the second-highest salary per department.
50. Explain the difference between INNER, LEFT, and FULL OUTER JOIN.
51. What's the difference between WHERE and HAVING?
52. Write a query using a window function to rank users by spend within each region.
53. Explain ROW_NUMBER, RANK and DENSE_RANK.
54. Write a query to compute 7-day rolling retention.
55. Write a query to find users who did A but not B.
56. How would you find duplicate rows and keep only the most recent?
57. Explain a self-join and give a use case.
58. Your query is slow on a billion-row table. How do you diagnose it?
59. What is a CTE and when does it help readability versus performance?
60. Compute month-over-month growth in SQL.

## A5. Product sense and metrics

61. How would you measure the success of a new feature?
62. Define a north-star metric for a customer-feedback analytics product.
63. What's the difference between a leading and a lagging indicator?
64. Daily active users dropped 5% yesterday. Walk me through your investigation.
65. How would you decide whether to build a feature at all?
66. How do you choose between two metrics that conflict?
67. What's a vanity metric? Give an example.
68. How would you measure the quality of a search results page?
69. How would you detect if a metric change is real or seasonal?
70. Design metrics for a customer support chatbot.
71. How would you quantify the business value of your complaint taxonomy?
72. A stakeholder wants a number that supports a decision they've already made. What do you do?

## A6. Modelling and machine learning for DS

73. How do you decide whether a problem needs machine learning at all?
74. Walk me through how you'd approach a churn prediction problem.
75. Your model has 99% accuracy on an imbalanced dataset. What do you check?
76. Explain precision, recall and when you'd optimise for each.
77. What is the ROC curve, and when is PR-AUC more informative?
78. Explain regularisation, L1 versus L2.
79. Explain how gradient boosting works. Why is it strong on tabular data?
80. How would you handle missing data? List three approaches and their assumptions.
81. Explain feature engineering choices you'd make for text data.
82. What is target leakage? Give an example that's easy to miss.
83. How do you explain a model's prediction to a non-technical stakeholder?
84. Explain SHAP values at a high level.
85. How do you decide when a model is good enough to deploy?
86. How would you monitor a model in production without ground-truth labels?
87. Your model performs well offline and badly online. List possible causes.
88. When would you choose an interpretable model over a more accurate one?

## A7. Case and communication

89. Estimate the number of customer reviews written in the UK per day.
90. A company wants to reduce customer churn by 10%. How do you approach it?
91. You have one week and messy data. How do you prioritise?
92. Present a negative result to a stakeholder who expected a positive one.
93. How would you explain a p-value to a marketing manager?
94. Tell me about an analysis that changed a decision.
95. Tell me about a time your analysis was wrong.
96. How do you handle a request for analysis you think is the wrong question?
97. Describe a dataset you found harder to work with than expected.
98. How do you decide what to put in a summary versus an appendix?

---

# PART B — AI ENGINEER

Your strongest fit. Most of this maps directly onto work you've already done.

## B1. LLM fundamentals

99. Explain how a decoder-only transformer generates text.
100. What is the difference between an encoder-only, decoder-only and encoder-decoder model?
101. Explain tokenisation. Why does it matter for cost and for non-English text?
102. What is a context window and what happens when you exceed it?
103. Explain temperature, top-p and top-k. When would you change each?
104. Why did you choose temperature 0.2 for your pipeline?
105. What is the KV cache and why does it matter for inference cost?
106. Explain why generation latency scales with output tokens but prefill scales with input tokens.
107. What causes hallucination, mechanically?
108. What is instruction tuning? What is RLHF?
109. Explain LoRA and why it's cheaper than full fine-tuning.
110. When would you fine-tune rather than prompt or retrieve?
111. What is a mixture-of-experts model and what problem does it solve?
112. How do open-weight and hosted models differ in what they let you control?

## B2. Prompting

113. What separates a good prompt from a bad one?
114. Explain few-shot prompting. When does it beat fine-tuning?
115. What is chain-of-thought and when does it not help?
116. How do you get reliable structured output from an LLM?
117. How would you enforce a JSON schema on model output?
118. What do you do when the model ignores an instruction?
119. Explain the difference between system and user messages.
120. How do you version and test prompts?
121. Your prompt works on 9 of 10 cases. How do you find the tenth before users do?
122. How would you A/B test a prompt change in production?

## B3. RAG

123. Explain RAG and the problem it solves.
124. Walk me through every failure mode of a RAG system.
125. When is RAG the wrong choice?
126. Explain chunking strategies and their trade-offs.
127. How do you choose chunk size and overlap?
128. What is late chunking? What is contextual retrieval?
129. Explain hybrid search. Why combine dense and sparse retrieval?
130. What is reciprocal rank fusion?
131. Where does a reranker go and what does it cost?
132. Explain query expansion and query decomposition.
133. How do you handle a question that needs information from twenty documents?
134. How do you cite sources so a user can verify a claim?
135. Design a RAG system over 100 million documents.
136. How do you keep an index fresh with continuously arriving data?
137. Explain your grounding threshold and how you'd calibrate it.
138. When should a RAG system refuse to answer?
139. How do you handle multi-turn conversations in RAG?
140. What is agentic RAG and when is the added complexity justified?
141. How would you support multi-tenant RAG with strict data isolation?

## B4. Embeddings and vector stores

142. How would you choose an embedding model?
143. What's the trade-off between embedding dimensionality and quality?
144. Explain the difference between symmetric and asymmetric search.
145. Why normalise embeddings?
146. Compare FAISS, Pinecone, Weaviate, Qdrant and pgvector.
147. Explain HNSW. What do M and efSearch control?
148. Explain product quantisation and what accuracy it costs.
149. How do you evaluate retrieval quality without an LLM?
150. What is Matryoshka representation learning?
151. How would you fine-tune an embedding model on domain data?
152. How do you handle documents in multiple languages?

## B5. Evaluation

153. How do you evaluate an LLM application when there's no single right answer?
154. Explain the RAGAS metrics and what each one isolates.
155. What are the weaknesses of LLM-as-judge?
156. What biases do LLM judges exhibit — position, length, self-preference?
157. How would you validate that your judge agrees with humans?
158. How do you build an evaluation set for a new LLM feature?
159. How many test cases do you need before a mean is meaningful?
160. What is a golden dataset and how do you keep it from going stale?
161. How do you evaluate a system that legitimately refuses some questions?
162. How do you catch a regression when you change a prompt?
163. Design an offline evaluation harness that runs in CI.
164. How do you measure hallucination rate in production, with no labels?
165. What online metrics would tell you your RAG system is degrading?

## B6. Safety, guardrails and reliability

166. What is prompt injection? Direct versus indirect.
167. Your retrieved documents are user-generated. How do you defend against injection?
168. Why is a prompt instruction a weak defence, and what's the strong version?
169. How would you prevent PII leaking into a third-party API?
170. How do you stop a model producing harmful or off-brand output?
171. Design a scope guard. What are the trade-offs of keyword versus classifier?
172. What happens to your guardrails when the LLM provider has an outage?
173. How would you rate-limit an LLM endpoint to bound cost?
174. How do you handle a model provider silently changing model behaviour?
175. What's your rollback strategy for a bad prompt deployment?

## B7. Agents and tool use

176. What is an agent, and when is one actually warranted?
177. Explain function calling and how the model decides which tool to use.
178. How do you handle a tool call that fails or times out?
179. How do you stop an agent looping indefinitely?
180. How do you evaluate an agent's trajectory rather than just its final answer?
181. What is the ReAct pattern?
182. How would you design human-in-the-loop approval for risky actions?
183. What's the difference between an agent and a well-structured pipeline — and which does your project use?

## B8. Cost, latency and serving

184. Break down where latency goes in your RAG pipeline.
185. How would you halve the cost of an LLM feature?
186. Explain semantic caching. What's the cache key and what invalidates it?
187. When would you route between a small and a large model?
188. What is speculative decoding?
189. Explain batching for LLM inference and its effect on latency.
190. How would you stream tokens, and what does it change for the user?
191. Self-hosted versus API — how do you decide?
192. What's the memory footprint of serving a 7B model, and how do you reduce it?
193. Explain quantisation and what it costs you.

## B9. Coding

194. Implement a token-budget-aware context builder that packs documents into a prompt.
195. Implement retry with backoff for a rate-limited API, retrying only what should be retried.
196. Implement a semantic cache with a similarity threshold.
197. Write a streaming endpoint that forwards LLM tokens to a client.
198. Implement chunking with overlap that respects sentence boundaries.
199. Parse and validate structured JSON output from an LLM, handling malformed responses.
200. Implement reciprocal rank fusion over two ranked lists.
201. Write a rate limiter (token bucket) for an LLM endpoint.
202. Implement cosine similarity and top-k without a vector library.
203. Write a function that redacts PII from text before it goes into a prompt.

## B10. System design

204. Design a customer support assistant over a company's knowledge base.
205. Design a system that summarises 10,000 documents a day.
206. Design a code-search assistant for a large monorepo.
207. Design an LLM feature with a hard 200ms p99 budget.
208. Design an evaluation and monitoring stack for an LLM product.
209. Design a system that keeps a vector index in sync with a live database.
210. Design multi-tenant RAG with per-tenant data isolation and per-tenant cost tracking.

## B11. Attacks on your work

211. Your RAG scored worse than the baseline. Why should I hire someone whose system lost?
212. You evaluated on five questions. That's not evidence.
213. Your judge is the same model that wrote the answers. That's not evaluation.
214. You wrote your own reference answers and then measured against them. Circular?
215. Your scope guard misses most out-of-scope questions. Would you ship that?
216. Your threshold is a hand-picked number. How is that engineering?
217. Everyone builds RAG chatbots now. What's hard about yours?

---

# HOW TO USE THIS

Answer out loud, timed. Mark green / amber / red. Work the reds.

**If you're targeting AI Engineer:** B3, B5 and B11 are where your project gives you real answers rather than textbook ones — lead with those. B6 and B7 are your thinnest areas; you have no agent work at all, so prepare B7 from reading rather than experience and say so honestly.

**If you're targeting Data Scientist:** A2 and A5 decide most loops. Your project gives you almost nothing there — that's a genuine preparation gap, not a delivery one. Do A4 (SQL) with a keyboard, not in your head.

**If you're unsure which to target:** your project is a much better fit for AI Engineer than for Data Scientist. The overlap with your actual evidence is roughly 70% versus 20%.
