"""Build a synthetic retrieval-evaluation dataset from the indexed notes.

For each sampled chunk, the local LLM generates one question whose answer is
contained in that chunk. The ground truth for the question is the chunk itself,
plus the adjacent chunks of the same paper (controlled by --include-neighbors).

The dataset is written as JSON Lines. Because the dataset is stored on disk,
every comparison run in evaluate_retrieval.py uses the exact same questions.

Usage:
    python scripts/build_eval_dataset.py --out eval/dataset.jsonl --seed 42
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.services.elasticsearch import es_service  # noqa: E402
from app.services.llm import llm_service  # noqa: E402

_CHUNK_RE = re.compile(r"^(.*)_chunk_(\d+)$")
_NOISE_RE = re.compile(r"(?i)^(references|bibliography|contents|works cited|index)\b")
_FALLBACK_PREFIX = "Failed to generate answer from local LLM"


def load_chunks() -> list[dict]:
    body = {"_source": ["chunk_id", "text", "paper_id", "category", "title"], "size": 10000}
    response = es_service.client.search(index=settings.INDEX_NAME, body=body)
    chunks = [hit["_source"] for hit in response["hits"]["hits"]]
    if len(chunks) >= 10000:
        print("warning: index holds 10000+ chunks; the sample may be incomplete", file=sys.stderr)
    return chunks


def _is_usable(text: str, min_len: int) -> bool:
    text = text.strip()
    return len(text) >= min_len and not _NOISE_RE.match(text)


def _chunk_index(chunk_id: str) -> int | None:
    match = _CHUNK_RE.match(chunk_id)
    return int(match.group(2)) if match else None


def sample_chunks(chunks: list[dict], count: int, per_paper: int, min_len: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    usable = [c for c in chunks if _is_usable(c.get("text") or "", min_len)]
    by_paper: dict[str, list[dict]] = {}
    for chunk in usable:
        by_paper.setdefault(chunk.get("paper_id") or "", []).append(chunk)

    selected: list[dict] = []
    for paper_chunks in by_paper.values():
        selected.extend(rng.sample(paper_chunks, min(per_paper, len(paper_chunks))))
    rng.shuffle(selected)
    return selected[:count]


def neighboring_chunk_ids(chunk_id: str, same_paper: list[str]) -> list[str]:
    idx = _chunk_index(chunk_id)
    if idx is None:
        return []
    return sorted(
        candidate
        for candidate in same_paper
        if (other := _chunk_index(candidate)) is not None and abs(other - idx) == 1
    )


def _clean_question(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\*\*|__|\*", "", text)
    text = re.sub(r"(?i)^(?:student )?question\s*:?\s*", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


async def generate_question(chunk: dict) -> str:
    text = chunk["text"].strip()
    prompt = (
        "Write exactly one concise question that a student might ask, such that the "
        "answer to the question is fully contained in the passage below. "
        "Output only the question, with no introduction.\n\n"
        f"Passage:\n{text}\n\n"
        "Question:"
    )
    raw = (await llm_service.generate_response(prompt, context_chunks=[])).strip()
    question = _clean_question(raw)
    if not question or _FALLBACK_PREFIX in question:
        return f"Summarize the paragraph that starts with: '{' '.join(text.split()[:10])}'"
    return question


async def build_dataset(
    chunks: list[dict],
    count: int,
    per_paper: int,
    min_len: int,
    seed: int,
    include_neighbors: bool,
    concurrency: int = 4,
) -> list[dict]:
    selected = sample_chunks(chunks, count, per_paper, min_len, seed)
    by_paper: dict[str, list[str]] = {}
    for chunk in chunks:
        by_paper.setdefault(chunk["paper_id"] or "", []).append(chunk["chunk_id"])

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded_question(chunk: dict) -> str:
        async with semaphore:
            return await generate_question(chunk)

    questions = await asyncio.gather(*(guarded_question(chunk) for chunk in selected))

    entries: list[dict] = []
    for i, chunk in enumerate(selected, start=1):
        question = questions[i - 1]
        relevant = [chunk["chunk_id"]]
        if include_neighbors:
            relevant += neighboring_chunk_ids(chunk["chunk_id"], by_paper.get(chunk["paper_id"] or "", []))
        entries.append(
            {
                "question": question,
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk.get("title", ""),
                "category": chunk.get("category", ""),
                "relevant": sorted(set(relevant)),
                "generated_from": "heuristic" if question.startswith("Summarize the paragraph") else "llm",
            }
        )
    return entries


def write_jsonl(entries: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


async def async_main(args: argparse.Namespace) -> int:
    print(f"Loading chunks from index '{settings.INDEX_NAME}' ...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    entries = await build_dataset(
        chunks,
        count=args.count,
        per_paper=args.per_paper,
        min_len=args.min_len,
        seed=args.seed,
        include_neighbors=args.include_neighbors,
        concurrency=args.concurrency,
    )
    if not entries:
        print("No usable chunks found. Widen the filters (--min-len).", file=sys.stderr)
        return 1

    write_jsonl(entries, Path(args.out))
    print(f"Wrote {len(entries)} questions to {args.out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a synthetic retrieval evaluation dataset.")
    parser.add_argument("--out", default="eval/dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--per-paper", type=int, default=2)
    parser.add_argument("--min-len", type=int, default=300)
    parser.add_argument("--include-neighbors", action="store_true", default=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())