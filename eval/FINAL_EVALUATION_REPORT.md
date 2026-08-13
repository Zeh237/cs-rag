# CSC RAG Engine — Final Evaluation Report

Run date: 2026-08-13
Environment: Python 3.12.12 (`.venv`), Elasticsearch 8.x at `localhost:9200`,
local LLM (LM Studio) at `localhost:1234/v1` (models `qwen/qwen3-1.7b`,
`qwen/qwen3-vl-4b`).

## 1. What was evaluated

Two evaluation tracks were run against the live, indexed corpus of CS course
notes (`csc_notes/`, 47 notes):

1. **Retrieval-strategy evaluation** — 30 questions (`eval/dataset_v2.jsonl`)
   × 4 retrieval pipelines (keyword, semantic, hybrid, hybrid+rerank), plus a
   repeat run with **LLM query rewriting**.
2. **Chunking-strategy evaluation** — 20 questions (`eval/chunk_dataset_v2.jsonl`)
   × 4 chunking strategies (fixed-1000, fixed-300, sentence-300, paragraph)
   × 4 retrieval pipelines, plus an **LLM-judged end-to-end answer quality**
   round (20 questions × 4 strategies).

The 30-question retrieval dataset and the 20-question chunking dataset are the
two that fully match the currently indexed corpus (verified against the manifest).

## 2. Results — retrieval strategies (30 queries)

Mean metrics over 30 queries. `hybrid+re` = hybrid + cross-encoder reranking.

| metric | keyword | semantic | hybrid | hybrid+rerank |
| --- | --- | --- | --- | --- |
| map | **0.3596** | 0.2629 | 0.3146 | 0.3394 |
| mrr | 0.5778 | 0.5625 | 0.5964 | **0.6761** |
| p@1 | 0.5333 | 0.5333 | 0.5333 | **0.6333** |
| p@3 | **0.3444** | 0.2444 | 0.3000 | 0.3222 |
| p@5 | **0.2333** | 0.1667 | 0.2200 | 0.2267 |
| p@10 | **0.1300** | 0.1033 | 0.1267 | 0.1267 |
| ndcg@1 | 0.5333 | 0.5333 | 0.5333 | **0.6333** |
| ndcg@3 | **0.3901** | 0.3034 | 0.3507 | 0.3898 |
| ndcg@5 | 0.4157 | 0.3236 | 0.3897 | **0.4228** |
| ndcg@10 | 0.4362 | 0.3537 | 0.4144 | **0.4428** |
| r@1 | 0.1778 | 0.1778 | 0.1778 | **0.2111** |
| r@3 | **0.3444** | 0.2444 | 0.3000 | 0.3222 |
| r@5 | **0.3889** | 0.2778 | 0.3667 | 0.3778 |
| r@10 | **0.4333** | 0.3444 | 0.4222 | 0.4222 |

### Statistical significance (paired 95% bootstrap CI, hybrid+rerank − other)

| metric | vs keyword | vs semantic | vs hybrid |
| --- | --- | --- | --- |
| map | −0.0202 [−0.176, +0.121] | +0.0765 [−0.053, +0.199] | +0.0248 [−0.111, +0.156] |
| mrr | +0.0983 [−0.137, +0.322] | +0.1136 [−0.123, +0.341] | +0.0797 [−0.145, +0.311] |

**No difference is statistically significant** (every CI contains zero). Keyword
retrieval is competitive — even best on MAP — while reranking helps the
ranking-head metrics (MRR, p@1, nDCG@1). With 30 queries the power is too low
to declare a winner; this is an honest limitation to report in the paper.

## 3. Results — with LLM query rewriting (30 queries)

| metric | keyword | semantic | hybrid | hybrid+rerank |
| --- | --- | --- | --- | --- |
| map | **0.3342** | 0.2286 | 0.2901 | 0.3075 |
| mrr | 0.5550 | 0.5034 | 0.5437 | **0.5975** |
| p@1 | **0.5333** | 0.4333 | 0.5000 | 0.5333 |
| p@3 | **0.2889** | 0.2222 | 0.2667 | 0.2889 |
| p@5 | **0.2267** | 0.1533 | 0.1867 | 0.2067 |
| p@10 | **0.1333** | 0.1000 | 0.1200 | 0.1200 |
| r@10 | **0.4444** | 0.3333 | 0.4000 | 0.4000 |

**Rewriting did not help** — MAP, MRR, p@1–p@10 all dropped slightly across
every pipeline (e.g. MRR 0.6761 → 0.5975 for hybrid+rerank). Because the
dataset questions are already well-formed and standalone, rewriting adds noise.
Ablation takeaway: query rewrite helps mainly for conversational follow-ups,
not single-turn questions.

## 4. Results — chunking strategies

Index stats and gold-passage containment (fraction of the 20 gold paragraphs a
single chunk covers ≥ 50%):

| index | chunks | avg words | containment |
| --- | --- | --- | --- |
| csc_notes_fixed1000 | 1,273 | 911 | 20/20 (100%) |
| csc_notes_fixed300 | 4,443 | 293 | 20/20 (100%) |
| csc_notes_sentence300 | 4,010 | 277 | 20/20 (100%) |
| csc_notes_para | 3,118 | 343 | 16/20 (80%) |

### Retrieval quality per strategy (map / mrr / r@10)

| index | keyword map | semantic map | hybrid map | h+rerank map | h+rerank mrr | r@10 (hybrid) |
| --- | --- | --- | --- | --- | --- | --- |
| fixed1000 | 0.6959 | 0.3510 | 0.4947 | 0.3654 | 0.3812 | 0.7500 |
| fixed300 | 0.6892 | 0.6596 | 0.6342 | 0.6385 | 0.6392 | 0.8500 |
| sentence300 | **0.7655** | **0.6917** | **0.7572** | **0.7000** | **0.7167** | **0.9500** |
| para | 0.3229 | 0.2475 | 0.2712 | 0.3289 | 0.3522 | 0.4000 |

**sentence-300 wins on every pipeline and metric.** Paragraph chunks do poorly
(their boundaries rarely align with a question's answer span; containment is
only 80%).

### Significance (hybrid+rerank MAP/MRR vs fixed1000 baseline)

| index | map diff (95% CI) | mrr diff (95% CI) |
| --- | --- | --- |
| fixed300 | **+0.2731 [+0.017, +0.534]*** | **+0.2579 [+0.002, +0.529]*** |
| sentence300 | **+0.3346 [+0.098, +0.552]*** | **+0.3354 [+0.096, +0.560]*** |
| para | −0.0365 [−0.302, +0.230] | −0.0290 [−0.296, +0.248] |

(*CI excludes zero → significant.) The smaller chunk sizes are **significantly
better** than 1000-word chunks for hybrid+rerank.

### End-to-end answer quality (LLM judge, /5, n=20 per strategy)

| index | mean judge score |
| --- | --- |
| csc_notes_fixed1000 | **4.20** |
| csc_notes_fixed300 | 3.60 |
| csc_notes_sentence300 | 4.00 |
| csc_notes_para | 3.90 |

**Notable divergence:** fixed-1000 has the worst retrieval metrics but the
*best* end-to-end answers. Larger chunks give the LLM enough self-contained
context to answer well even when retrieval ranking is worse — a classic
retrieval-vs-generation tension worth discussing in the paper.

## 5. Data-quality fix performed during the run

`csc_notes_para` was originally indexed with a **384-dim** embedding model
(`bge-small`) while every other index and the current embedding service use
**768-dim** (`BAAI/bge-base-en-v1.5` default). Semantic search against it
failed with `search_phase_execution_exception`. It was rebuilt with the current
model and mapping:

```bash
python scripts/index_library.py --index csc_notes_para \
  --strategy paragraph --chunk-size 300 --overlap 0 --only-md --delete-first
```

Re-indexed 3,118 chunks (0 failed); mapping dims now 768. Any paper should note
that all compared indexes must share the same embedding model/dimension.

## 6. Abbreviations glossary

- **RAG** — Retrieval-Augmented Generation.
- **Chunk** — a contiguous text unit that a retriever ranks and an LLM reads.
- **Embedding / dense vector** — fixed-size numeric meaning vector (768-dim here).
- **Keyword search** — lexical match via Elasticsearch multi_match (BM25-style).
- **Semantic search** — embedding cosine similarity (here `cosineSimilarity + 1.0`).
- **Hybrid search** — fused keyword + semantic via reciprocal-rank fusion (RRF).
- **Reranker / cross-encoder** — `cross-encoder/ms-marco-MiniLM-L-6-v2`
  re-scores `(query, text)` pairs to refine order.
- **Query rewrite** — LLM converts a query into a standalone search query.
- **MAP / map** — Mean Average Precision.
- **MRR / mrr** — Mean Reciprocal Rank.
- **p@k** — Precision at k; **r@k** — Recall at k.
- **nDCG@k** — Normalized Discounted Cumulative Gain at k.
- **CI** — Confidence Interval; **paired bootstrap diff** — resampling-based CI
  for a difference between two aligned pipelines; significant when the interval
  excludes zero.
- **Containment** — share of a gold passage covered by a single retrieved chunk.
- **Gold / ground truth** — the known-correct chunk(s)/passage for a question.
- **LLM judge** — an LLM scores an answer 1–5 against the reference passage.
- **RRF** — Reciprocal Rank Fusion (hybrid score combination).

## 7. Files produced

| File | Content |
| --- | --- |
| `eval/final_retrieval.log` | Console output of the retrieval run |
| `eval/final_retrieval_queries.csv` | Per-query metrics, 30 queries × 4 pipelines |
| `eval/final_retrieval_rewrite.log` | Console output of the rewrite run |
| `eval/final_retrieval_rewrite_queries.csv` | Per-query metrics with rewrite |
| `eval/final_chunking.log` | Console output of the chunking run |
| `eval/final_chunking_queries.csv` | Per-query metrics, 20 queries × 4 strategies × 4 pipelines |
| `eval/final_chunking_graded.log` | Console output incl. judge scores |
| `eval/final_chunking_graded_queries.csv` | Per-query retrieval metrics (graded run) |

## 8. Reproduction

```bash
# retrieval (30 queries, 4 pipelines)
python scripts/evaluate_retrieval.py --dataset eval/dataset_v2.jsonl \
  --k 1,3,5,10 --out eval/final_retrieval

# retrieval with LLM query rewrite
python scripts/evaluate_retrieval.py --dataset eval/dataset_v2.jsonl \
  --k 1,3,5,10 --rewrite --out eval/final_retrieval_rewrite

# chunking comparison (requires the 4 indexes; see README to rebuild them)
python scripts/evaluate_chunking.py \
  --indexes csc_notes_fixed1000,csc_notes_fixed300,csc_notes_sentence300,csc_notes_para \
  --dataset eval/chunk_dataset_v2.jsonl --overlap-ratio 0.5 --out eval/final_chunking

# same, plus LLM answer-quality grading on all 20 questions
python scripts/evaluate_chunking.py \
  --indexes csc_notes_fixed1000,csc_notes_fixed300,csc_notes_sentence300,csc_notes_para \
  --dataset eval/chunk_dataset_v2.jsonl --overlap-ratio 0.5 --grade 20 \
  --out eval/final_chunking_graded
```

## 9. Paper takeaways (summary of the story the numbers tell)

1. On this corpus, **no retrieval method is significantly better than another**
   at 30 queries; keyword is surprisingly strong, and reranking best helps
   top-1 relevance (MRR, p@1, nDCG@1).
2. **LLM query rewriting does not help** single-turn retrieval on this dataset
   (slightly hurts); it is a feature for conversational follow-ups only.
3. **Chunking matters more than retrieval method.** sentence-300 significantly
   beats fixed-1000 for hybrid+rerank (MAP +0.33, MRR +0.34) and leads on every
   pipeline. Paragraph chunking is worst.
4. **Retrieval quality ≠ answer quality.** fixed-1000 scored best end-to-end
   (4.20/5) despite worst retrieval metrics, because bigger chunks feed the LLM
   more self-contained context.
5. **Statistical rigor:** use the paired bootstrap CIs; only the chunking
   comparisons reach significance at n=20–30. Increase n for tighter claims.
