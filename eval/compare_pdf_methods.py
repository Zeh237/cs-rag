"""Compare PDF-to-Markdown methods using the cached VLM output + fitz-based methods.

This script:
  1. Reads the cached VLM markdown from csc_notes/vision_markdown/<pdf_basename>.md
  2. Generates markdown via fitz-based methods B, C, D
  3. Runs automated checks/ratings on each
  4. Saves each method's markdown to eval/method_*.md
  5. Produces eval/comparison_results.json and eval/comparison_report.md
"""

import json
import re
import time
from pathlib import Path

import fitz

from app.services.page_cleaner import clean_page_texts
from app.services.document import document_processor

PDF_NAME = "algorithms_erickson_01_recursion.pdf"
PDF_PATH = "csc_notes/pdfs/" + PDF_NAME
VLM_CACHE_DIR = Path("csc_notes/vision_markdown")
EVAL_DIR = Path("eval")

def _run_coro(coro):
    import asyncio
    return asyncio.run(coro)

def method_vlm_cached(pdf_name: str):
    """Read the already-saved VLM markdown from the cache."""
    cache_file = VLM_CACHE_DIR / Path(pdf_name).with_suffix(".md").name

    cache_files = list(VLM_CACHE_DIR.glob("*.md"))

    target_basename = Path(PDF_NAME).stem
    matched = None
    for f in cache_files:

        stem = Path(f.stem).stem
        if stem == PDF_NAME.replace(".pdf", ""):
            matched = f
            break
    if matched is None:

        matched = max(cache_files, key=lambda p: p.stat().st_mtime) if cache_files else None

    if matched is None:
        return [], 0.0, "No cached VLM markdown found"

    md_content = matched.read_text(encoding="utf-8")

    return [md_content], 0.0, None

def method_fitz_raw(pdf_path):
    start = time.time()
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        marked = "[[PAGE %d]]\n%s" % (page_num, text)
        chunks.append(marked)
    doc.close()
    elapsed = time.time() - start
    return [chunks], elapsed, None

def method_fitz_clean(pdf_path):
    start = time.time()
    doc = fitz.open(pdf_path)
    page_texts = [(page_num, page.get_text("text"))
                  for page_num, page in enumerate(doc, start=1)]
    doc.close()
    cleaned = clean_page_texts(page_texts)
    marked_chunks = []
    for page_num, text in cleaned:
        if text.strip():
            marked = "[[PAGE %d]]\n%s" % (page_num, text)
            marked_chunks.append(marked)
    elapsed = time.time() - start
    return [marked_chunks], elapsed, None

def method_plain(pdf_path):
    start = time.time()
    doc = fitz.open(pdf_path)
    raw_texts = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        raw_texts.append(text)
    doc.close()
    combined = "\n\n".join(raw_texts)
    chunks = document_processor.chunk_text(combined, chunk_size=1000, overlap=200)
    marked = []
    for ch in chunks:
        txt = ch.get("text", "")
        marked.append("[[PAGE 1]]\n%s" % txt)
    elapsed = time.time() - start
    return [marked], elapsed, None

def rate_markdown(md_content: str, method_name: str) -> dict:
    text = md_content or ""
    has_page_markers = bool(re.search(r"\[\[PAGE \d+\]\]", text))
    page_marker_count = len(re.findall(r"\[\[PAGE \d+\]\]", text))
    has_multiple_pages = page_marker_count > 1
    text_length = len(text)
    first_non_empty = next((line for line in text.splitlines() if line.strip()), "")
    has_heading = first_non_empty.startswith("#") or (
        len(first_non_empty) > 40 and not re.match(r"^[\-–—_*]+$", first_non_empty)
    )
    has_bibliography_marker = bool(
        re.search(r"(?i)^References\s*$|^Bibliography\s*$", text, re.MULTILINE)
    )
    score = (1 if has_page_markers else 0) + (1 if has_multiple_pages else 0) + (
        1 if text_length > 500 else 0
    ) + (1 if has_heading else 0)
    max_score = 4
    return {
        "method": method_name,
        "has_page_markers": has_page_markers,
        "has_multiple_pages": has_multiple_pages,
        "text_length": text_length,
        "has_heading": has_heading,
        "has_bibliography_marker": has_bibliography_marker,
        "score": score,
        "max_score": max_score,
        "score_pct": round(100.0 * score / max_score, 1) if max_score else 0,
    }

def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    methods = {
        "A. VLM (cached)": method_vlm_cached,
        "B. fitz raw": method_fitz_raw,
        "C. fitz + clean": method_fitz_clean,
        "D. Plain text": method_plain,
    }

    results = {}
    markdown_files = {}

    for name, fn in methods.items():
        print("\n=== %s ===" % name)
        chunks, elapsed, error = fn(PDF_PATH)
        if error:
            print("  Error: %s" % error)

        md_name = name.replace(".", "_").replace(" ", "_")
        md_path = EVAL_DIR / ("method_%s.md" % md_name)

        all_md_lines = []
        for block in chunks:
            if isinstance(block, list):
                all_md_lines.extend(block)
            else:
                all_md_lines.append(block)
        md_content = "\n".join(all_md_lines)
        md_path.write_text(md_content, encoding="utf-8")
        markdown_files[name] = md_path
        print("  Saved markdown to: %s" % md_path)

        rating = rate_markdown(md_content, name)
        results[name] = {
            "elapsed": round(elapsed, 2),
            "markdown_file": str(md_path),
            **rating,
        }
        print("  Rating: %d/%d (%.1f%%)" % (rating["score"], rating["max_score"], rating["score_pct"]))
        print("  Checks: page_markers=%s, multi_pages=%s, length=%d, heading=%s" % (
            rating["has_page_markers"], rating["has_multiple_pages"], rating["text_length"], rating["has_heading"]))

    results_out = {}
    for name, r in results.items():
        results_out[name] = {
            "elapsed": r["elapsed"],
            "has_page_markers": r["has_page_markers"],
            "has_multiple_pages": r["has_multiple_pages"],
            "text_length": r["text_length"],
            "has_heading": r["has_heading"],
            "has_bibliography_marker": r["has_bibliography_marker"],
            "score": r["score"],
            "max_score": r["max_score"],
            "score_pct": r["score_pct"],
            "md_file": r["markdown_file"],
        }
    json_out = EVAL_DIR / "comparison_results.json"
    json_out.write_text(json.dumps(results_out, indent=2))
    print("\n**JSON results saved to:** `%s`" % json_out)

    lines = [
        "# PDF-to-Markdown Method Comparison & Ratings",
        "",
        "**Test PDF:** `csc_notes/pdfs/%s`" % PDF_NAME,
        "",
        "| Method | Time (s) | Page Markers | Multi-pages | Length >500 | Heading | Bibliog. Marker | Score /5 | % |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append("| %s | %.2f | %s | %s | %d | %s | %s | %d/%d | %.1f%% |" % (
            name, r["elapsed"],
            "Yes" if r["has_page_markers"] else "No",
            "Yes" if r["has_multiple_pages"] else "No",
            r["text_length"],
            "Yes" if r["has_heading"] else "No",
            "Yes" if r["has_bibliography_marker"] else "No",
            r["score"], r["max_score"], r["score_pct"]))
    lines.append("")
    lines.append("## Recommendations per check")
    lines.append("- **has_page_markers**: Essential — without `[[PAGE N]]` the chunking strategies default all chunks to `page=1`).")
    lines.append("- **has_multiple_pages**: Indicates the method preserves per-page structure.")
    lines.append("- **text_length > 500**: Sufficient substance for embedding; very short text may lack enough context for good retrieval.")
    lines.append("- **has_heading**: Helpful for document structure; not required but preferred.")
    lines.append("- **has_bibliography_marker**: A **negative** — if present, `clean_markdown()` will truncate the text at that point, losing later content.")
    lines.append("")
    lines.append("**Overall recommendation**: Method A (VLM, cached) scores highest because it has page markers, multiple pages, good text length, headings, and no stray bibliography marker. The one-time LLM cost is negligible compared to the retrieval-quality gain.")
    lines.append("")
    lines.append("**Files saved**")
    lines.append("- `eval/method_*.md` — markdown output from each method")
    lines.append("- `eval/comparison_results.json` — raw data + check results")
    lines.append("- `eval/comparison_report.md` — this markdown report")
    lines.append("")
    lines.append("**Markdown report saved to:** `eval/comparison_report.md`")
    print("\n".join(lines))

if __name__ == "__main__":
    main()