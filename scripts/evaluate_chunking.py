"""Compare retrieval quality across chunking strategies.

Builds several Elasticsearch indexes from the same notes with different
chunking strategies (see scripts/index_library.py --strategy), then evaluates
the same chunking-agnostic question set against each index using the shared
word-span coordinate system. A retrieved chunk is relevant when its stored
word span overlaps the gold answer paragraph by at least --overlap-ratio.

Metrics reuse app/services/evaluation.py so the tables are directly comparable
to the retrieval-method evaluation.

Usage:
    python scripts/index_library.py --index csc_notes_fixed300 \\
        --strategy fixed --chunk-size 300 --overlap 45 --only-md
    python scripts/evaluate_chunking.py \\
        --indexes csc_notes_fixed1000,csc_notes_fixed300,csc_notes_sentence300,csc_notes_para \\
        --dataset eval/chunk_dataset.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunking import span_overlap_fraction  # noqa: E402
from app.services.elasticsearch import es_service  # noqa: E402
from app.services.evaluation import (  # noqa: E402
    average_precision,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.embedding import embedding_service  # noqa: E402
from app.services.llm import llm_service  # noqa: E402

PIPELINES = ["keyword", "semantic", "hybrid", "hybrid+rerank"]


def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def center_overlap_fraction(hit: dict, gold_start: int, gold_end: int) -> float:
    h_start = hit.get("span_start")
    h_end = hit.get("span_end")
    if h_start is None or h_end is None:
        return 0.0
    return span_overlap_fraction(h_start, h_end, gold_start, gold_end)


def is_relevant(hit: dict, gold_start: int, gold_end: int, ratio: float) -> bool:
    return center_overlap_fraction(hit, gold_start, gold_end) >= ratio


async def retrieve(index: str, question: str, top_k: int) -> dict[str, list[dict]]:
    query_vector = embedding_service.get_embeddings(question)
    keyword = es_service.keyword_search(question, limit=top_k, index=index)
    semantic = es_service.semantic_search_cosine(query_vector, limit=top_k, index=index)
    hybrid = es_service.hybrid_search(question, query_vector, limit=top_k, rerank=False, index=index)
    hybrid_rerank = es_service.hybrid_search(question, query_vector, limit=top_k, rerank=True, index=index)
    return {
        "keyword": keyword,
        "semantic": semantic,
        "hybrid": hybrid,
        "hybrid+rerank": hybrid_rerank,
    }


def entry_metrics(ranks_by_pipeline: dict[str, list[dict]], entry: dict, ks: list[int], ratio: float) -> dict:
    gold_start, gold_end = entry["span_start"], entry["span_end"]
    output = {}
    for pipeline, hits in ranks_by_pipeline.items():
        ranks = [h["chunk_id"] for h in hits]
        relevant = {h["chunk_id"] for h in hits if is_relevant(h, gold_start, gold_end, ratio)}
        metrics = {
            "mrr": reciprocal_rank(ranks, relevant),
            "map": average_precision(ranks, relevant),
        }
        for k in ks:
            metrics[f"p@{k}"] = precision_at_k(ranks, relevant, k)
            metrics[f"r@{k}"] = recall_at_k(ranks, relevant, k)
        output[pipeline] = metrics
    return output


def metrics_names(ks: list[int]) -> list[str]:
    names = ["map", "mrr"]
    for k in ks:
        names += [f"p@{k}", f"r@{k}"]
    return names


def index_stats_spans(index: str) -> list[tuple[str, int, int]]:
    body = {
        "_source": ["span_start", "span_end", "paper_id"],
        "query": {"match_all": {}},
        "size": 10000,
    }
    response = es_service.client.search(index=index, body=body)
    spans = []
    for hit in response["hits"]["hits"]:
        source = hit["_source"]
        start = source.get("span_start")
        end = source.get("span_end")
        if start is not None and end is not None:
            spans.append((source["paper_id"], int(start), int(end)))
    return spans


def best_single_chunk_overlap(spans: list[tuple[str, int, int]], entry: dict) -> float:
    gold_len = entry["span_end"] - entry["span_start"]
    if gold_len <= 0:
        return 0.0
    best = 0.0
    for paper, start, end in spans:
        if paper != entry["paper_id"]:
            continue
        intersection = max(0, min(end, entry["span_end"]) - max(start, entry["span_start"]))
        best = max(best, intersection / gold_len)
    return best


def print_table(collected: dict[str, dict[str, dict[str, list[float]]]], ks: list[int], indexes: list[str]) -> None:
    shown = ["map", "mrr"] + [f"r@{k}" for k in ks] + [f"p@{k}" for k in ks]
    for pipeline in PIPELINES:
        print(f"\nPipeline: {pipeline}")
        header = "index".ljust(24) + "".join(m.ljust(10) for m in shown)
        print(header)
        print("-" * len(header))
        for index in indexes:
            row = index[:23].ljust(24)
            for metric in shown:
                values = collected[index][pipeline][metric]
                row += f"{mean(values):.4f}".ljust(10)
            print(row)
    print()


def print_significance(collected: dict[str, dict[str, dict[str, list[float]]]], baseline: str) -> None:
    print(f"Paired 95% bootstrap CIs for the difference vs {baseline} (hybrid+rerank):")
    for metric in ["map", "mrr"]:
        for index in collected:
            if index == baseline:
                continue
            mean_diff, low, high = paired_bootstrap_diff(
                collected[index]["hybrid+rerank"][metric],
                collected[baseline]["hybrid+rerank"][metric],
                seed=42,
            )
            significant = low > 0 or high < 0
            marker = "*" if significant else ""
            print(f"  {metric:6s} {index:24s} diff={mean_diff:+.4f} [{low:+.4f}, {high:+.4f}]{marker}")
    print("  (*) the interval excludes zero, so the difference is significant.\n")


def write_csv(entries: list[dict], collected: dict[str, dict[str, dict[str, list[float]]]], out_prefix: str) -> None:
    out_path = Path(out_prefix + "_queries.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["question", "paper_id", "span_start", "span_end"]
    for index in collected:
        for pipeline in PIPELINES:
            for metric in collected[index][pipeline]:
                columns.append(f"{index}:{pipeline}:{metric}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        n = len(entries)
        for q_index, entry in enumerate(entries):
            row = [
                json.dumps(entry["question"], ensure_ascii=False).replace('"', '""'),
                entry.get("paper_id", ""),
                str(entry["span_start"]),
                str(entry["span_end"]),
            ]
            for index in collected:
                for pipeline in PIPELINES:
                    for metric in collected[index][pipeline]:
                        row.append(f"{collected[index][pipeline][metric][q_index]:.4f}")
            f.write(",".join(row) + "\n")
    print(f"Per-query results written to {out_path}")


async def build_collected(
    entries: list[dict], indexes: list[str], ks: list[int], ratio: float, top_k: int
) -> dict[str, dict[str, dict[str, list[float]]]]:
    collected = {index: {pipeline: {m: [] for m in metrics_names(ks)} for pipeline in PIPELINES} for index in indexes}
    for position, entry in enumerate(entries, start=1):
        q = entry["question"]
        for index in indexes:
            ranks = await retrieve(index, q, top_k)
            metrics = entry_metrics(ranks, entry, ks, ratio)
            for pipeline in PIPELINES:
                for metric in metrics_names(ks):
                    collected[index][pipeline][metric].append(metrics[pipeline][metric])
        if position % 20 == 0:
            print(f"  evaluated {position}/{len(entries)}")
    return collected


async def grade_answers(
    entries: list[dict],
    indexes: list[str],
    limit: int,
    top_k: int,
) -> None:
    if limit <= 0:
        return
    print(f"Grading answers for {limit} questions per index (this is slow)...")
    scores: dict[str, list[float]] = {index: [] for index in indexes}
    for entry in entries[:limit]:
        gold = entry["passage"]
        for index in indexes:
            hits = (await retrieve(index, entry["question"], top_k))["hybrid+rerank"]
            context = [
                {"title": h.get("title") or h.get("paper_id") or "Notes", "chunk_id": h.get("chunk_id"), "text": h["text"]}
                for h in hits
            ]
            answer = await llm_service.generate_response(entry["question"], context_chunks=context)
            score = await judge_score(entry["question"], gold, answer)
            scores[index].append(score)
            print(f"  [{entry['question'][:50]}...] {index}: {score:.1f}/5")
    print("\nAnswer quality (mean judge score /5):")
    for index in indexes:
        values = scores[index]
        print(f"  {index:<28} {mean(values):.2f}  (n={len(values)})")
    print()


async def judge_score(question: str, gold_passage: str, answer: str) -> float:
    prompt = (
        "You are grading the correctness of an answer to a question, given the "
        "passage that contains the correct answer.\n\n"
        f"Question: {question}\n\n"
        f"Reference passage:\n{gold_passage}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Score the answer from 1 (wrong or unrelated) to 5 (fully correct and "
        "faithful to the passage). Reply with only a number 1-5."
    )
    raw = (await llm_service.generate_response(prompt, context_chunks=[])).strip()
    try:
        number = float(raw.split()[0])
    except (ValueError, IndexError):
        return 1.0
    return min(5.0, max(1.0, number))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def async_main(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 1
    entries = load_dataset(dataset_path)
    if args.quick is not None:
        entries = entries[: args.quick]
    if not entries:
        print("The dataset is empty.", file=sys.stderr)
        return 1

    indexes = [i.strip() for i in args.indexes.split(",") if i.strip()]
    ks = [int(k) for k in args.k.split(",") if k]
    top_k = max(ks)

    print("index stats:")
    span_cache: dict[str, list[tuple[str, int, int]]] = {}
    for index in indexes:
        spans = index_stats_spans(index)
        span_cache[index] = spans
        sizes = [end - start for _, start, end in spans]
        avg = sum(sizes) / len(sizes) if sizes else 0.0
        print(f"  {index:<28} {len(spans):>6} chunks  avg {avg:.0f} words.")

    collected = await build_collected(entries, indexes, ks, args.overlap_ratio, top_k)

    print(f"\nGold passage containment (a single chunk covers >= {args.overlap_ratio:.0%} of it):")
    for index in indexes:
        covered = sum(
            1 for e in entries if best_single_chunk_overlap(span_cache[index], e) >= args.overlap_ratio
        )
        pct = 100.0 * covered / len(entries)
        print(f"  {index:<28} {covered:>{len(str(len(entries)))}d}/{len(entries)} ({pct:.1f}%)")

    print_table(collected, ks, indexes)
    print_significance(collected, args.baseline)

    if args.grade:
        await grade_answers(entries, indexes, args.grade, top_k)

    if args.out_prefix:
        write_csv(entries, collected, args.out_prefix)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval quality across chunking strategies.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--indexes", required=True, help="Comma-separated Elasticsearch index names.")
    parser.add_argument("--baseline", default="csc_notes_fixed1000")
    parser.add_argument("--k", default="1,3,5,10")
    parser.add_argument("--overlap-ratio", type=float, default=0.5)
    parser.add_argument("--quick", type=int, default=None)
    parser.add_argument("--grade", type=int, default=0, help="Sample size for LLM answer-quality grading.")
    parser.add_argument("--out-prefix", default="eval/chunk_results")
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())