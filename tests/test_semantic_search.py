import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from nlp.embedding_service import semantic_search


print("=" * 80)
print("Testing Semantic Search")
print("=" * 80)

query = "battery drains quickly"

results = semantic_search(
    query=query,
    top_k=5
)

results_rating = semantic_search(
    query="battery drains quickly",
    top_k=5,
    min_rating=3
)

results_negative = semantic_search(
    query="battery drains quickly",
    top_k=5,
    sentiment_filter="negative"
)

print(f"\nQuery: {query}")
print(f"Retrieved {len(results)} reviews\n")

for i, review in enumerate(results, start=1):
    print(f"Result {i}")
    print(f"Similarity : {review['similarity_percentage']}")
    print(f"Platform   : {review['platform']}")
    print(f"Rating     : {review['rating']}")
    print(f"Sentiment  : {review['sentiment_label']}")
    print(f"Review     : {review['text']}")
    print("-" * 80)