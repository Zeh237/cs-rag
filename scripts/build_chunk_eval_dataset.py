"""Build a chunking-agnostic retrieval evaluation dataset from the notes.

Unlike build_eval_dataset.py, which generates ground truth from a specific
index's chunks, this script samples *paragraphs* directly from the raw notes
and records their word-span in the shared coordinate system produced by
app.services.chunking.parse_document. Indexers and the chunking evaluator use
the same parser, so a returned chunk can be judged relevant by word-index
overlap with the gold paragraph instead of by chunk ids. This makes the dataset
usable across chunking strategies.

Usage:
    python scripts/build_chunk_eval_dataset.py --notes-dir ./csc_notes \
        --out eval/chunk_dataset.jsonl --count 100 --seed 42
"""

import argparse
import asyncio
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402
from app.services.chunking import parse_document  # noqa: E402
from app.services.llm import llm_service  # noqa: E402

_FALLBACK_PREFIX = "Failed to generate answer from local LLM"


@dataclass
class Passage:
    paper_id: str
    heading: str
    start: int
    end: int
    text: str


def load_notes(notes_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(notes_dir.glob("*.md")):
        files[path.stem] = path.read_text(encoding="utf-8", errors="ignore")
    return files


def collect_passages(notes: dict[str, str], min_words: int) -> list[Passage]:
    passages: list[Passage] = []
    for paper_id, text in notes.items():
        doc = parse_document(text)
        if not doc.words:
            continue
        for section in doc.sections:
            for paragraph in section.paragraphs:
                size = paragraph.end - paragraph.start
                if size < min_words:
                    continue
                passage = " ".join(doc.words[paragraph.start : paragraph.end]).strip()
                passages.append(
                    Passage(paper_id, section.heading, paragraph.start, paragraph.end, passage)
                )
    return passages


def sample_passages(passages: list[Passage], count: int, per_book: int, seed: int) -> list[Passage]:
    rng = random.Random(seed)
    by_book: dict[str, list[Passage]] = {}
    for passage in passages:
        book = passage.paper_id.split("_")[0]
        by_book.setdefault(book, []).append(passage)

    selected: list[Passage] = []
    for book_passages in by_book.values():
        selected.extend(rng.sample(book_passages, min(per_book, len(book_passages))))
    rng.shuffle(selected)
    return selected[:count]


def _clean_question(raw: str) -> str:
    text = raw.strip()
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = _FALLBACK_PREFIX + " " if _FALLBACK_PREFIX in text else text
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


async def generate_question(passage: Passage) -> str:
    prompt = (
        "Write exactly one concise question that a student might ask, such that the "
        "answer to the question is fully contained in the passage below. "
        "Output only the question, with no introduction.\n\n"
        f"Passage:\n{passage.text}\n\n"
        "Question:"
    )
    raw = (await llm_service.generate_response(prompt, context_chunks=[])).strip()
    question = _clean_question(raw)
    if not question or _FALLBACK_PREFIX in question:
        return f"Summarize the paragraph that starts with: '{' '.join(passage.text.split()[:10])}'"
    return question


async def build_dataset(passages: list[Passage], concurrency: int = 4) -> list[dict]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(passage: Passage) -> str:
        async with semaphore:
            return await generate_question(passage)

    questions = await asyncio.gather(*(guarded(p) for p in passages))

    entries = []
    for i, passage in enumerate(passages, start=1):
        question = questions[i - 1]
        entries.append(
            {
                "question": question,
                "paper_id": passage.paper_id,
                "heading": passage.heading,
                "span_start": passage.start,
                "span_end": passage.end,
                "passage": passage.text,
                "generated_from": "heuristic" if question.startswith("Summarize the paragraph") else "llm",
            }
        )
    return entries


def write_jsonl(entries: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chunking-agnostic retrieval evaluation dataset.")
    parser.add_argument("--notes-dir", default=settings.NOTES_DIR)
    parser.add_argument("--out", default="eval/chunk_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--per-book", type=int, default=4)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    notes = load_notes(Path(args.notes_dir))
    passages = collect_passages(notes, args.min_words)
    print(f"Loaded {len(notes)} notes, {len(passages)} usable paragraphs.")

    selected = sample_passages(passages, args.count, args.per_book, args.seed)
    if not selected:
        print("No passages matched the filters.", file=sys.stderr)
        return

    entries = asyncio.run(build_dataset(selected, args.concurrency))
    write_jsonl(entries, Path(args.out))
    print(f"Wrote {len(entries)} questions to {args.out}")


if __name__ == "__main__":
    main()