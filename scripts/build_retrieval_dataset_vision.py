"""Build retrieval dataset from vision_markdown MD files.

Extracts clean, prose-rich passages directly from the vision markdown, detects
real section headings (both "# Markdown" and plain "1.2 Title" styles), skips
noise (code blocks, formula/math fragments, figure captions, Latin filler,
page markers, footnotes), and generates a natural question per passage with the
LLM. The LLM receives the section heading as context so questions read like a
student's question. Bad or missing LLM output falls back to a heading-based
question.

Usage:
    python scripts/build_retrieval_dataset_vision.py --out eval/retrieval_dataset_vision.jsonl --count 40 --seed 42
"""

import argparse
import asyncio
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.llm import llm_service  # noqa: E402

_FALLBACK_PREFIX = "Failed to generate answer from local LLM"

_PAGE_RE = re.compile(r"^\[\[PAGE (\d+)\]\]$")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$")
_PLAIN_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*|(?:[A-ZА-Я]\.)+)\s+[A-ZА-Я]", re.UNICODE)
_FIGURE_RE = re.compile(r"^\s*(figure|fig\.?|algorithm|table|example)\b", re.IGNORECASE)
_FOOTNOTE_RE = re.compile(r"^\d{1,3}[)\]]$")
_LATIN_WORDS = {
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "proin", "quis", "augue", "ligula", "primvs", "dignitas", "intamtenviscientia",
    "esseresenimsuntarvaepropeinsingvlis", "atqveinterpvnc", "bonorum",
}
_CODE_WORD_HINTS = {"return", "printf", "func", "function", "def ", "for", "while", "input",
                    "prev", "curr", "next", "dist", "if", "then", "else", ":= ", "←"}
_MATH_WORD_HINTS = {"sum", "prod", "sqrt", "infty", "over", "cdot", "frac", "displaystyle", "ell"}


@dataclass
class Passage:
    paper_id: str
    source_file: str
    heading: str
    start: int
    end: int
    text: str
    page_start: int
    page_end: int


def load_vision_notes(vision_dir: Path, fallback_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    vision_files = sorted(vision_dir.glob("*.md")) if vision_dir.is_dir() else []
    if vision_files:
        for p in vision_files:
            stem = p.stem
            paper_id = stem.rsplit("_", 1)[0] if "_" in stem and len(stem.rsplit("_", 1)[-1]) == 12 else stem
            files[paper_id] = p.read_text(encoding="utf-8", errors="ignore")
        print(f"Loaded {len(files)} vision markdown files from {vision_dir}")
    else:
        for p in sorted(fallback_dir.glob("*.md")):
            files[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
        print(f"Loaded {len(files)} fallback files from {fallback_dir}")
    return files


def _strip_md(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~`>#]", "", text)
    return text


def _is_latin(text: str) -> bool:
    words = {w.strip(".,;:!?").lower() for w in text.split()}
    if len(words) < 4:
        return False
    marked = sum(1 for w in words if w in _LATIN_WORDS)
    return marked / len(words) >= 0.3


def _is_code_or_math(text: str, math_chars: int) -> bool:
    words = text.split()
    if not words:
        return False
    dense_code = sum(1 for w in words if w in _CODE_WORD_HINTS)
    if dense_code >= max(3, len(words) // 4):
        return True
    dense_math = sum(1 for w in words if w in _MATH_WORD_HINTS)
    if dense_math >= 2:
        return True
    if math_chars > 0 and math_chars / len(text) > 0.15:
        return True
    if any(ch in text for ch in "{}=+") and len(re.findall(r"[A-Z]\[[0-9\. ]+", text)) > 0:
        return True
    return False


def _looks_like_prose(tokens: list[str]) -> bool:
    text = " ".join(tokens)
    if is_caption := _FIGURE_RE.match(text):
        return False
    if _is_latin(text):
        return False
    math_chars = text.count("∑") + text.count("Δ") + text.count("∞") + text.count("_") + text.count("→") + text.count("•") + text.count("−")
    if _is_code_or_math(text, math_chars):
        return False
    if len(tokens) < 12:
        return False
    return True


def _collect_v2(notes: dict[str, str], min_words: int) -> list[Passage]:
    passages: list[Passage] = []
    for paper_id, text in notes.items():
        source_file = paper_id
        current_heading = "Introduction"
        current_page = 1
        tokens: list[str] = []
        token_pages: list[int] = []
        heading_boundaries: list[tuple[int, str]] = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            page_match = _PAGE_RE.match(stripped)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            heading_match = _HEADING_RE.match(stripped)
            plain_match = None
            if not heading_match:
                plain_match = _PLAIN_HEADING_RE.match(stripped)
                # Reject exercise/body lines that merely start with a number: real
                # headings are short titles (e.g. "3.7 Edit Distance").
                if plain_match and len(stripped.split()) > 12:
                    plain_match = None
            if heading_match or plain_match:
                h = _strip_md(heading_match.group(1) if heading_match else stripped).strip()
                heading_boundaries.append((len(tokens), h))
                current_heading = h
                continue
            body = _strip_md(stripped)
            if not body.strip():
                continue
            toks = body.split()
            tokens.extend(toks)
            token_pages.extend([current_page] * len(toks))

        groups: list[tuple[str, int, int]] = []
        for i, (idx, h) in enumerate(heading_boundaries):
            end = heading_boundaries[i + 1][0] if i + 1 < len(heading_boundaries) else len(tokens)
            groups.append((h, idx, end))
        if not groups:
            groups.append(("Introduction", 0, len(tokens)))

        for heading, g_start, g_end in groups:
            for para_start, para_end in _split_paragraphs(tokens, g_start, g_end):
                size = para_end - para_start
                if size < min_words:
                    continue
                para_tokens = tokens[para_start:para_end]
                if not _looks_like_prose(para_tokens):
                    continue
                passage_text = " ".join(para_tokens).strip()
                if not passage_text:
                    continue
                page_start = token_pages[para_start]
                page_end = token_pages[para_end - 1]
                passages.append(
                    Passage(paper_id, source_file, heading, para_start, para_end, passage_text, page_start, page_end)
                )
    return passages


def _split_paragraphs(tokens: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) into sentence-bounded passages of a few sentences each."""
    if start >= end:
        return []
    max_len = 90
    # Identify sentence end token indices (token followed by '.'/'?'/'!')
    sent_ends: list[int] = []
    for i in range(start, end - 1):
        tok = tokens[i]
        if tok and tok[-1] in ".?!":
            sent_ends.append(i + 1)
    spans: list[tuple[int, int]] = []
    begin = start
    # Anchor on sentence ends so each passage has full sentences.
    seg = start
    while seg < end:
        nxt = seg + max_len
        if nxt >= end:
            nxt = end
            spans.append((seg, nxt))
            break
        # find nearest sentence end >= seg+max_len*0.6 and <= nxt
        candidates = [se for se in sent_ends if seg + int(max_len * 0.6) <= se <= nxt]
        if candidates:
            nxt = candidates[-1]
        elif sent_ends and seg + max_len // 2 < end:
            near = [se for se in sent_ends if seg + max_len // 2 <= se <= seg + max_len * 2]
            if near:
                nxt = near[0]
        spans.append((seg, nxt))
        seg = nxt
    return spans


def sample_passages(passages: list[Passage], count: int, per_book: int, seed: int) -> list[Passage]:
    rng = random.Random(seed)
    by_book: dict[str, list[Passage]] = {}
    for p in passages:
        book = p.paper_id.split("_")[0]
        by_book.setdefault(book, []).append(p)
    selected: list[Passage] = []
    for book_passages in by_book.values():
        selected.extend(rng.sample(book_passages, min(per_book, len(book_passages))))
    rng.shuffle(selected)
    return selected[:count]


def _clean_question(raw: str) -> str:
    text = raw.strip().replace("**", "").replace("__", "").replace("*", "")
    if _FALLBACK_PREFIX in text:
        text = ""
    text = text.replace("\n", " ").replace("\r", " ")
    return " ".join(text.split()).strip()


_QUESTION_TEMPLATES = [
    "What is {topic}?",
    "Can you explain {topic}?",
    "How does {topic} work?",
    "What are the main ideas behind {topic}?",
    "Why is {topic} important?",
    "Describe {topic}.",
    "What is the significance of {topic}?",
]


def _heading_based_question(heading: str) -> str:
    topic = heading.strip()
    # Drop leading numeric chapter/section labels to get a readable topic.
    if re.match(r"^[\d\s.]+[A-ZА-Я]", topic):
        m = re.match(r"^[\d\s.]+(.*)$", topic)
        topic = (m.group(1) if m else topic).strip()
    topic = topic.strip(":")
    if not topic:
        topic = heading.strip()
    seed = hash(topic) % (2**32)
    rng = random.Random(seed)
    tmpl = rng.choice(_QUESTION_TEMPLATES)
    return tmpl.format(topic=topic)


async def generate_question(passage: Passage, llm_call) -> str:
    prompt = (
        "You are writing exam-style questions for a computer science study helper. "
        "A student wants to look up a concept in algorithms course notes. "
        "Write exactly ONE short, concise question (10 words or fewer) such that "
        "the answer is fully contained in the passage below. Vary the question "
        "style: sometimes ask \"What is X?\", sometimes \"How does X work?\", "
        "sometimes \"Why is X important?\" or \"How does X relate to Y?\". Never "
        "quote raw text or code from the passage verbatim. Output ONLY the "
        "question, nothing else.\n\n"
        f"Section heading: {passage.heading}\n"
        f"Passage:\n{passage.text}\n\n"
        "Question:"
    )
    raw = await llm_call(prompt)
    q = _clean_question(raw)
    # Reject LLM output that just re-quotes fragments or is degenerate
    if not q or q.startswith(("What does the passage say", "Summarize the paragraph")):
        q = _heading_based_question(passage.heading)
    # Drop leading fragments that look like quoting
    q = re.sub(r"^(what is|explain|describe|how does|why is|what are|how would you|can you)\s+a\s+", "", q, flags=re.IGNORECASE)
    # Heuristic: if the question lacks a real verb and looks like a raw paste, fall back
    if len(q.split()) < 3:
        q = _heading_based_question(passage.heading)
    if q and not q.endswith(("?", ".")):
        q += "?"
    return q


async def generate_answer(passage: Passage, question: str, llm_call) -> str:
    prompt = (
        "Answer the following question in 2-3 clear sentences, using ONLY the "
        "passage below as the source. Do not add outside information. Output "
        "only the answer.\n\n"
        f"Question: {question}\n\n"
        f"Passage:\n{passage.text}\n\n"
        "Answer:"
    )
    raw = await llm_call(prompt)
    a = _clean_question(raw)
    if not a or _FALLBACK_PREFIX in raw:
        return passage.text
    return a


async def generate_question_answer(passage: Passage, llm_call) -> tuple[str, str]:
    question = await generate_question(passage, llm_call)
    answer = await generate_answer(passage, question, llm_call)
    return question, answer


async def build_dataset(passages: list[Passage], concurrency: int = 2) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def llm_call(prompt: str) -> str:
        async with sem:
            try:
                return (await llm_service.generate_response(prompt, context_chunks=[])).strip()
            except Exception:
                return ""

    async def build_one(p: Passage) -> tuple[str, str]:
        return await generate_question_answer(p, llm_call)

    outputs = await asyncio.gather(*(build_one(p) for p in passages))
    entries = []
    for passage, (question, answer) in zip(passages, outputs):
        heuristic = _heading_based_question(passage.heading)
        is_heuristic = question == heuristic
        entries.append({
            "question": question,
            "answer": answer,
            "paper_id": passage.paper_id,
            "source_file": passage.source_file,
            "heading": passage.heading,
            "span_start": passage.start,
            "span_end": passage.end,
            "passage": passage.text,
            "page_start": passage.page_start,
            "page_end": passage.page_end,
            "generated_from": "heuristic" if is_heuristic else "llm",
        })
    return entries


def write_jsonl(entries: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval dataset from vision_markdown.")
    parser.add_argument("--vision-dir", default="csc_notes/vision_markdown")
    parser.add_argument("--fallback-dir", default="csc_notes")
    parser.add_argument("--out", default="eval/retrieval_dataset_vision.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--per-book", type=int, default=40)
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    notes = load_vision_notes(Path(args.vision_dir), Path(args.fallback_dir))
    passages = _collect_v2(notes, args.min_words)
    print(f"Collected {len(passages)} usable prose passages (min-words={args.min_words})")
    if not passages:
        print("No passages matched filters.", file=sys.stderr)
        sys.exit(1)

    selected = sample_passages(passages, args.count, args.per_book, args.seed)
    print(f"Sampled {len(selected)} passages")
    if not selected:
        print("Sampling produced 0 entries.", file=sys.stderr)
        sys.exit(1)

    entries = asyncio.run(build_dataset(selected, args.concurrency))
    write_jsonl(entries, Path(args.out))
    print(f"Wrote {len(entries)} questions to {args.out}")


if __name__ == "__main__":
    main()
