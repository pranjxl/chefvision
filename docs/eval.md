Self-retrieval benchmark: 200 queries, 60% of ingredients revealed, backend 'sentence-transformers/all-MiniLM-L6-v2', weights Weights(coverage=0.45, usage=0.2, similarity=0.2, missing=0.15).

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | ms/query |
|---|---|---|---|---|---|
| Coverage only (naive) | 0.840 | 1.000 | 1.000 | 0.912 | 86.3 |
| Overlap (coverage + usage + missing) | 0.915 | 1.000 | 1.000 | 0.958 | 85.4 |
| Semantic only | 0.495 | 0.730 | 0.830 | 0.604 | 212.4 |
| Hybrid (all signals) | 0.960 | 1.000 | 1.000 | 0.980 | 99.5 |

Proxy metric: the target recipe is in the corpus; ranks beyond 50 count as misses.
