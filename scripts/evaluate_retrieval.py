"""Compare retrieval strategies on the synthetic evaluation dataset.

The runner evaluates keyword-only, semantic-only, hybrid, and
hybrid-with-reranker search. Every retriever goes through the same service
methods that the app uses, so the numbers reflect the production ranking code.

Usage:
    python scripts/evaluate_retrieval.py --dataset eval/dataset.jsonl \
        --k 1,3,5,10 --out eval/results
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.elasticsearch import es_service  # noqa: E402
from app.services.embedding import embedding_service  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    average_precision,
    bootstrap_ci,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.llm import llm_service  # noqa: E402

VARIANTS = ["keyword", "semantic", "hybrid", "hybrid+rerank"]


def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _ranks(hits: list[dict]) -> list[str]:
    return [hit["chunk_id"] for hit in hits]


async def retrieve_question(question: str, top_k: int, use_rewrite: bool) -> dict[str, list[str]]:
    query = question
    if use_rewrite:
        query = await llm_service.rewrite_query_for_search(question, None)

    query_vector = embedding_service.get_query_embeddings(query)

    keyword = _ranks(es_service.keyword_search(query, limit=top_k))[:top_k]
    semantic = _ranks(es_service.semantic_search_cosine(query_vector, limit=top_k))[:top_k]
    hybrid = _ranks(es_service.hybrid_search(query=query, query_vector=query_vector, limit=top_k, rerank=False))
    hybrid_rerank = _ranks(es_service.hybrid_search(query=query, query_vector=query_vector, limit=top_k, rerank=True))

    return {
        "keyword": keyword,
        "semantic": semantic,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rerank,
    }


def evaluate_query(ranks_by_variant: dict[str, list[str]], relevant: set[str], ks: list[int]) -> dict:
    metrics = {}
    for variant, ranks in ranks_by_variant.items():
        entry = {}
        for k in ks:
            entry[f"p@{k}"] = precision_at_k(ranks, relevant, k)
            entry[f"r@{k}"] = recall_at_k(ranks, relevant, k)
            entry[f"ndcg@{k}"] = ndcg_at_k(ranks, relevant, k)
        entry["mrr"] = reciprocal_rank(ranks, relevant)
        entry["map"] = average_precision(ranks, relevant)
        metrics[variant] = entry
    return metrics


def summarize_metrics(per_question: list[tuple[dict, dict]], ks: list[int]) -> dict:
    summaries = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        for metric in ["map", "mrr"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks] + [f"ndcg@{k}" for k in ks]:
            values = [metrics[variant][metric] for _, metrics in per_question]
            mean, low, high = bootstrap_ci(values, seed=42)
            summaries[variant][metric] = {"mean": mean, "low": low, "high": high}
    return summaries


def print_table(summaries: dict, ks: list[int]) -> None:
    shown = ["map", "mrr"] + [f"p@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + [f"r@{k}" for k in ks]

    header = "metric".ljust(10) + "".join(variant[:9].ljust(10) for variant in VARIANTS)
    print(header)
    print("-" * len(header))
    for metric in shown:
        row = metric.ljust(10)
        for variant in VARIANTS:
            row += f"{summaries[variant][metric]['mean']:.4f}".ljust(10)
        print(row)
    print()


def print_significance(per_question: list[tuple[dict, dict]]) -> None:
    baseline = "hybrid+rerank"

    def values_for(metric: str, variant: str) -> list[float]:
        return [metrics[variant][metric] for _, metrics in per_question]

    print("Paired 95% bootstrap CIs for the difference (hybrid+rerank - other):")
    for metric in ["map", "mrr"]:
        for other in ["keyword", "semantic", "hybrid"]:
            mean, low, high = paired_bootstrap_diff(
                values_for(metric, baseline), values_for(metric, other), seed=42
            )
            significant = low > 0 or high < 0
            marker = "*" if significant else ""
            print(f"  {metric:6s} vs {other:12s} diff={mean:+.4f} [{low:+.4f}, {high:+.4f}]{marker}")
    print("  (*) the interval excludes zero, so the difference is significant.\n")


def write_csv(per_question: list[tuple[dict, dict]], ks: list[int], out_prefix: str) -> None:
    out_path = Path(out_prefix + "_queries.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["question", "paper_id", "chunk_id"]
    for variant in VARIANTS:
        for metric in ["map", "mrr"] + [f"p@{k}" for k in ks] + [f"r@{k}" for k in ks]:
            columns.append(f"{variant}:{metric}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for entry, metrics in per_question:
            row = [
                json.dumps(entry["question"], ensure_ascii=False).replace('"', '""'),
                entry.get("paper_id", ""),
                entry.get("chunk_id", ""),
            ]
            for variant in VARIANTS:
                for metric in sorted(metrics[variant].keys()):
                    row.append(f"{metrics[variant][metric]:.4f}")
            f.write(",".join(row) + "\n")
    print(f"Per-query results written to {out_path}")


async def async_main(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    entries = load_dataset(dataset_path)
    if not entries:
        print("The dataset is empty.", file=sys.stderr)
        return 1
    if args.quick is not None:
        entries = entries[: args.quick]

    ks = [int(k) for k in args.k.split(",") if k]
    top_k = max(ks)

    per_question = []
    for position, entry in enumerate(entries, start=1):
        ranks = await retrieve_question(entry["question"], top_k, args.rewrite)
        per_question.append((entry, evaluate_query(ranks, set(entry["relevant"]), ks)))
        if position % 20 == 0:
            print(f"  evaluated {position}/{len(entries)}")

    summaries = summarize_metrics(per_question, ks)
    print_table(summaries, ks)
    print_significance(per_question)
    write_csv(per_question, ks, args.out_prefix)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval strategies on the eval dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--out-prefix", default="eval/results")
    parser.add_argument("--rewrite", action="store_true")
    parser.add_argument("--quick", type=int, default=None)
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())