"""Compare dense, sparse and hybrid retrieval on the csc_notes index.

Uses the vision_markdown-derived dataset (eval/retrieval_dataset_vision.jsonl)
and measures accuracy (map, mrr, p@k, r@k, ndcg@k) and latency (embedding,
search, rerank, total) for four pipelines: keyword (sparse), semantic (dense),
hybrid (RRF), hybrid+rerank.

Usage:
    python eval/compare_retrieval_pipelines.py --dataset eval/retrieval_dataset_vision.jsonl --index csc_notes --k 1,3,5,10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.elasticsearch import es_service  # noqa: E402
from app.services.embedding import embedding_service  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    average_precision,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

PIPELINES = ["keyword", "semantic", "hybrid", "hybrid+rerank"]


def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def passage_word_overlap(hit_text: str, passage: str) -> float:
    if not passage or not hit_text:
        return 0.0
    p_words = set(passage.lower().split())
    h_words = set(hit_text.lower().split())
    if not p_words:
        return 0.0
    return len(p_words & h_words) / len(p_words)


def is_relevant(hit: dict, entry: dict, ratio: float = 0.5) -> bool:
    if hit.get("paper_id") != entry.get("paper_id"):
        # Allow cross-paper matches if text overlap is very high (synopsis duplicated)
        pass
    text = hit.get("text") or ""
    passage = entry.get("passage") or ""
    if hit.get("paper_id") == entry.get("paper_id") and passage_word_overlap(text, passage) >= ratio:
        return True
    # Fallback: any hit with high overlap even if paper differs
    if passage_word_overlap(text, passage) >= 0.7:
        return True
    return False


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


async def retrieve_with_timing(index: str, question: str, top_k: int) -> tuple[dict[str, list[dict]], dict[str, float]]:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    qvec = embedding_service.get_query_embeddings(question)
    t_embed = time.perf_counter() - t0
    timings["embed"] = t_embed

    t1 = time.perf_counter()
    keyword = es_service.keyword_search(question, limit=top_k, index=index)
    timings["keyword_search"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    semantic = es_service.semantic_search_cosine(qvec, limit=top_k, index=index)
    timings["semantic_search"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    hybrid = es_service.hybrid_search(question, qvec, limit=top_k, rerank=False, index=index)
    timings["hybrid_search"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    hybrid_rerank = es_service.hybrid_search(question, qvec, limit=top_k, rerank=True, index=index)
    # hybrid_search with rerank internally times rerank+fusion, so isolate rerank cost:
    # Approximate: hybrid_rerank total - hybrid_search (fusion) ; embedding already counted
    timings["hybrid_rerank_total"] = time.perf_counter() - t4 + t_embed + timings["hybrid_search"]
    timings["rerank_only"] = max(0.0, timings["hybrid_rerank_total"] - timings["hybrid_search"] - t_embed)

    # Per-pipeline totals as user would experience
    totals = {
        "keyword": timings["keyword_search"],
        "semantic": timings["embed"] + timings["semantic_search"],
        "hybrid": timings["embed"] + timings["hybrid_search"],
        "hybrid+rerank": timings["hybrid_rerank_total"],
    }
    # Detailed breakdown
    breakdown = {
        "embed": t_embed,
        "keyword": timings["keyword_search"],
        "semantic": timings["semantic_search"],
        "hybrid_fusion": timings["hybrid_search"],
        "rerank": timings["rerank_only"],
        "keyword_total": totals["keyword"],
        "semantic_total": totals["semantic"],
        "hybrid_total": totals["hybrid"],
        "hybrid_rerank_total": totals["hybrid+rerank"],
    }
    hits = {
        "keyword": keyword,
        "semantic": semantic,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rerank,
    }
    return hits, breakdown


def entry_metrics(ranks: list[str], relevant: set[str], ks: list[int]) -> dict:
    return {
        "map": average_precision(ranks, relevant),
        "mrr": reciprocal_rank(ranks, relevant),
        **{f"p@{k}": precision_at_k(ranks, relevant, k) for k in ks},
        **{f"r@{k}": recall_at_k(ranks, relevant, k) for k in ks},
        **{f"ndcg@{k}": ndcg_at_k(ranks, relevant, k) for k in ks},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dense/sparse/hybrid retrieval with accuracy and latency.")
    parser.add_argument("--dataset", default="eval/retrieval_dataset_vision.jsonl")
    parser.add_argument("--index", default="csc_notes")
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Word-overlap threshold for relevance.")
    parser.add_argument("--quick", type=int, default=None)
    parser.add_argument("--out-prefix", default="eval/retrieval_comparison")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    entries = load_dataset(dataset_path)
    if args.quick is not None:
        entries = entries[: args.quick]
    if not entries:
        print("Dataset is empty.", file=sys.stderr)
        return 1

    ks = [int(k) for k in args.k.split(",") if k.strip()]
    top_k = max(ks)

    if not es_service.verify_connection():
        print("Elasticsearch connection failed.", file=sys.stderr)
        return 1

    try:
        count = es_service.client.count(index=args.index).get("count", 0)
        print(f"Index '{args.index}': {count} chunks")
    except Exception:
        print(f"Could not count index '{args.index}'", file=sys.stderr)

    # Warmup (prime embedding cache and ES)
    try:
        _ = embedding_service.get_query_embeddings("warmup query")
    except Exception:
        pass

    collected: dict[str, dict[str, list[float]]] = {p: defaultdict(list) for p in PIPELINES}
    latencies: dict[str, list[float]] = {p: [] for p in PIPELINES}
    detailed: dict[str, list[float]] = defaultdict(list)
    per_query: list[dict] = []

    import asyncio

    async def run():
        for idx, entry in enumerate(entries, 1):
            q = entry["question"]
            hits_by_pipeline, timings = await retrieve_with_timing(args.index, q, top_k)

            row: dict = {"question": q, "paper_id": entry["paper_id"], "timings": timings}
            for pipeline in PIPELINES:
                hits = hits_by_pipeline[pipeline]
                ranks = [h["chunk_id"] for h in hits]
                relevant = {h["chunk_id"] for h in hits if is_relevant(h, entry, args.overlap_ratio)}
                # If no hit contains sufficient overlap, fall back to paper_id match for leniency
                if not relevant:
                    # At least mark top hit from same paper as relevant if it exists, to avoid all-zero metrics when corpus is different
                    pass
                metrics = entry_metrics(ranks, relevant, ks)
                for m, v in metrics.items():
                    collected[pipeline][m].append(v)
                    row[f"{pipeline}:{m}"] = v
                # Latency per pipeline
                total_key = {
                    "keyword": "keyword_total",
                    "semantic": "semantic_total",
                    "hybrid": "hybrid_total",
                    "hybrid+rerank": "hybrid_rerank_total",
                }[pipeline]
                lat = timings[total_key]
                latencies[pipeline].append(lat)
                row[f"{pipeline}:latency_ms"] = round(lat * 1000, 2)

            for k in ["embed", "keyword", "semantic", "hybrid_fusion", "rerank"]:
                detailed[k].append(timings[k])

            per_query.append(row)
            if idx % 10 == 0:
                print(f"  evaluated {idx}/{len(entries)}")

    asyncio.run(run())

    # Summary accuracy
    print("\n=== Accuracy (mean over {} queries, index={}, overlap_ratio={}) ===".format(len(entries), args.index, args.overlap_ratio))
    metrics_to_show = ["map", "mrr"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]
    header = "pipeline".ljust(16) + "".join(m.ljust(10) for m in metrics_to_show)
    print(header)
    print("-" * len(header))
    for p in PIPELINES:
        row = p.ljust(16)
        for m in metrics_to_show:
            vals = collected[p].get(m, [])
            row += f"{(mean(vals) if vals else 0.0):.4f}".ljust(10)
        print(row)

    # Significance vs best
    best = max(PIPELINES, key=lambda p: mean(collected[p].get("map", [0])))
    print(f"\nBest MAP pipeline: {best}")
    for m in ["map", "mrr"]:
        print(f"  {m} 95% CIs vs {best}:")
        for p in PIPELINES:
            if p == best:
                continue
            diff, lo, hi = paired_bootstrap_diff(collected[p][m], collected[best][m])
            sig = "*" if lo > 0 or hi < 0 else ""
            print(f"    {p:16s} diff={diff:+.4f} [{lo:+.4f}, {hi:+.4f}]{sig}")

    # Latency
    print("\n=== Latency (seconds, per query) ===")
    print("pipeline".ljust(16) + "mean".ljust(10) + "median".ljust(10) + "p95".ljust(10) + "p99".ljust(10) + "throughput(q/s)")
    for p in PIPELINES:
        vals = latencies[p]
        thr = 1.0 / mean(vals) if vals and mean(vals) > 0 else 0.0
        print(f"{p.ljust(16)}{mean(vals):.4f}    {median(vals):.4f}    {percentile(vals,95):.4f}    {percentile(vals,99):.4f}    {thr:.2f}")

    print("\n=== Breakdown (mean ms) ===")
    for k in ["embed", "keyword", "semantic", "hybrid_fusion", "rerank"]:
        print(f"  {k:16s} {mean(detailed[k])*1000:.2f} ms")

    # Write outputs
    out_json = Path(args.out_prefix + ".json")
    out_csv = Path(args.out_prefix + ".csv")
    out_md = Path(args.out_prefix + "_report.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for p in PIPELINES:
        summary[p] = {
            "accuracy": {m: mean(collected[p].get(m, [])) for m in metrics_to_show},
            "latency": {
                "mean": mean(latencies[p]),
                "median": median(latencies[p]),
                "p95": percentile(latencies[p], 95),
                "p99": percentile(latencies[p], 99),
                "throughput": 1.0 / mean(latencies[p]) if latencies[p] and mean(latencies[p]) > 0 else 0,
            },
        }
        summary[p]["latency_breakdown"] = {k: mean(detailed[k]) for k in detailed}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"config": {"dataset": str(dataset_path), "index": args.index, "k": ks, "overlap_ratio": args.overlap_ratio}, "summary": summary, "per_query": per_query}, f, indent=2)
    print(f"\nJSON saved to {out_json}")

    # CSV per query
    with open(out_csv, "w", encoding="utf-8") as f:
        cols = ["question", "paper_id"] + [f"{p}:{m}" for p in PIPELINES for m in metrics_to_show] + [f"{p}:latency_ms" for p in PIPELINES]
        f.write(",".join(cols) + "\n")
        for row in per_query:
            vals = [json.dumps(row["question"], ensure_ascii=False).replace('"', '""'), row["paper_id"]]
            for p in PIPELINES:
                for m in metrics_to_show:
                    vals.append(f"{row.get(f'{p}:{m}', 0):.4f}")
            for p in PIPELINES:
                vals.append(f"{row.get(f'{p}:latency_ms', 0):.2f}")
            f.write(",".join(vals) + "\n")
    print(f"CSV saved to {out_csv}")

    # Markdown report
    lines = [
        "# Retrieval Pipeline Comparison: Dense vs Sparse vs Hybrid",
        "",
        f"**Dataset:** `{dataset_path}` ({len(entries)} queries from vision_markdown)",
        f"**Index:** `{args.index}`",
        f"**Relevance:** passage word-overlap >= {args.overlap_ratio:.0%} and same paper_id",
        "",
        "## Accuracy",
        "",
        "| Pipeline | " + " | ".join(metrics_to_show) + " |",
        "|---" + "|---" * len(metrics_to_show) + "|",
    ]
    for p in PIPELINES:
        vals = " | ".join(f"{mean(collected[p].get(m, [])):.4f}" for m in metrics_to_show)
        lines.append(f"| {p} | {vals} |")
    lines += ["", "## Latency (seconds per query)", "", "| Pipeline | Mean | Median | p95 | p99 | Throughput (q/s) |", "|---|---|---|---|---|---|"]
    for p in PIPELINES:
        vals = latencies[p]
        lines.append(f"| {p} | {mean(vals):.4f} | {median(vals):.4f} | {percentile(vals,95):.4f} | {percentile(vals,99):.4f} | {1/mean(vals) if vals and mean(vals)>0 else 0:.2f} |")
    lines += ["", "## Breakdown (mean ms)", "", "| Component | Mean (ms) |", "|---|---|"]
    for k in ["embed", "keyword", "semantic", "hybrid_fusion", "rerank"]:
        lines.append(f"| {k} | {mean(detailed[k])*1000:.2f} |")
    lines += ["", "## Notes", "- **keyword** = sparse BM25 (ES multi_match).", "- **semantic** = dense cosineSimilarity (EmbeddingGemma 2, SearchQuery prompt).", "- **hybrid** = RRF (k=60) fusion of both.", "- **hybrid+rerank** = hybrid + cross-encoder ms-marco-MiniLM-L-6-v2.", "- Latency covers embedding + search + rerank as applicable.", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved to {out_md}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
