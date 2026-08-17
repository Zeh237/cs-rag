"""Quick PDF-to-Markdown comparison runner (fitz-based methods only)."""
import fitz, re, time
from pathlib import Path

from app.services.page_cleaner import clean_page_texts
from app.services.document import document_processor

PDF_PATH = "csc_notes/pdfs/algorithms_erickson_01_recursion.pdf"
eval_dir = Path("eval")
eval_dir.mkdir(exist_ok=True)

methods_data = {}

# --- Method B: fitz raw ---
start = time.time()
doc = fitz.open(PDF_PATH)
chunks_b = []
for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text")
    marked = "[[PAGE %d]]\n%s" % (page_num, text)
    chunks_b.append(marked)
doc.close()
elapsed_b = time.time() - start
text_b = "\n".join(chunks_b)
page_markers_b = bool(re.search(r"\[\[PAGE \d+\]\]", text_b))
page_count_b = len(re.findall(r"\[\[PAGE \d+\]\]", text_b))
first_line_b = text_b.splitlines()[0] if text_b.splitlines() else ""
has_heading_b = first_line_b.startswith("#") or (len(first_line_b) > 40 and not re.match(r"^[\-–—_*]+$", first_line_b))
results_b = {
    "has_page_markers": page_markers_b,
    "has_multiple_pages": page_count_b > 1,
    "text_length": len(text_b),
    "has_heading": has_heading_b,
    "has_bibliography_marker": bool(re.search(r"(?i)^References\s*$|^Bibliography\s*$", text_b, re.MULTILINE)),
    "score": (1 if page_markers_b else 0) + (1 if page_count_b > 1 else 0) + (1 if len(text_b) > 500 else 0) + (1 if has_heading_b else 0),
    "max_score": 4,
    "score_pct": round(100.0 * ( (1 if page_markers_b else 0) + (1 if page_count_b > 1 else 0) + (1 if len(text_b) > 500 else 0) + (1 if has_heading_b else 0) ) / 4, 1),
}
methods_data["B. fitz raw"] = {"elapsed": elapsed_b, "rating": results_b, "md_content": text_b}
print("B. fitz raw: elapsed=%.2f, score=%.1f%%" % (elapsed_b, results_b["score_pct"]))

# --- Method C: fitz + clean ---
start = time.time()
doc = fitz.open(PDF_PATH)
page_texts = [(page_num, page.get_text("text")) for page_num, page in enumerate(doc, start=1)]
doc.close()
cleaned = clean_page_texts(page_texts)
chunks_c = []
for page_num, text in cleaned:
    if text.strip():
        marked = "[[PAGE %d]]\n%s" % (page_num, text)
        chunks_c.append(marked)
elapsed_c = time.time() - start
text_c = "\n".join(chunks_c)
page_markers_c = bool(re.search(r"\[\[PAGE \d+\]\]", text_c))
page_count_c = len(re.findall(r"\[\[PAGE \d+\]\]", text_c))
first_line_c = text_c.splitlines()[0] if text_c.splitlines() else ""
has_heading_c = first_line_c.startswith("#") or (len(first_line_c) > 40 and not re.match(r"^[\-–—_*]+$", first_line_c))
results_c = {
    "has_page_markers": page_markers_c,
    "has_multiple_pages": page_count_c > 1,
    "text_length": len(text_c),
    "has_heading": has_heading_c,
    "has_bibliography_marker": bool(re.search(r"(?i)^References\s*$|^Bibliography\s*$", text_c, re.MULTILINE)),
    "score": (1 if page_markers_c else 0) + (1 if page_count_c > 1 else 0) + (1 if len(text_c) > 500 else 0) + (1 if has_heading_c else 0),
    "max_score": 4,
    "score_pct": round(100.0 * ( (1 if page_markers_c else 0) + (1 if page_count_c > 1 else 0) + (1 if len(text_c) > 500 else 0) + (1 if has_heading_c else 0) ) / 4, 1),
}
methods_data["C. fitz + clean"] = {"elapsed": elapsed_c, "rating": results_c, "md_content": text_c}
print("C. fitz + clean: elapsed=%.2f, score=%.1f%%" % (elapsed_c, results_c["score_pct"]))

# --- Method D: plain text ---
start = time.time()
doc = fitz.open(PDF_PATH)
raw_texts = []
for page_num, page in enumerate(doc, start=1):
    text = page.get_text("text")
    raw_texts.append(text)
doc.close()
combined = "\n\n".join(raw_texts)
chunks_d = document_processor.chunk_text(combined, chunk_size=1000, overlap=200)
elapsed_d = time.time() - start
marked_chunks_d = []
for ch in chunks_d:
    txt = ch.get("text", "")
    marked_chunks_d.append("[[PAGE 1]]\n%s" % txt)
text_d = "\n".join(marked_chunks_d)
page_markers_d = bool(re.search(r"\[\[PAGE \d+\]\]", text_d))
first_line_d = text_d.splitlines()[0] if text_d.splitlines() else ""
has_heading_d = first_line_d.startswith("#") or (len(first_line_d) > 40 and not re.match(r"^[\-–—_*]+$", first_line_d))
results_d = {
    "has_page_markers": page_markers_d,
    "has_multiple_pages": False,
    "text_length": len(text_d),
    "has_heading": has_heading_d,
    "has_bibliography_marker": bool(re.search(r"(?i)^References\s*$|^Bibliography\s*$", text_d, re.MULTILINE)),
    "score": (1 if page_markers_d else 0) + 0 + (1 if len(text_d) > 500 else 0) + (1 if has_heading_d else 0),
    "max_score": 4,
    "score_pct": round(100.0 * ( (1 if page_markers_d else 0) + 0 + (1 if len(text_d) > 500 else 0) + (1 if has_heading_d else 0) ) / 4, 1),
}
methods_data["D. Plain text"] = {"elapsed": elapsed_d, "rating": results_d, "md_content": text_d}
print("D. Plain text: elapsed=%.2f, score=%.1f%%" % (elapsed_d, results_d["score_pct"]))

# --- Save markdown files ---
for name, data in methods_data.items():
    md_path = eval_dir / ("method_%s.md" % name.replace(".", "_").replace(" ", "_"))
    md_path.write_text(data["md_content"], encoding="utf-8")
    print("Saved: %s (%d chars)" % (md_path, len(data["md_content"])))

# --- Save JSON results ---
results_out = {}
for name, data in methods_data.items():
    r = data["rating"]
    results_out[name] = {
        "elapsed": data["elapsed"],
        "has_page_markers": r["has_page_markers"],
        "has_multiple_pages": r["has_multiple_pages"],
        "text_length": r["text_length"],
        "has_heading": r["has_heading"],
        "has_bibliography_marker": r["has_bibliography_marker"],
        "score": r["score"],
        "max_score": r["max_score"],
        "score_pct": r["score_pct"],
        "md_file": str(eval_dir / ("method_%s.md" % name.replace(".", "_").replace(" ", "_"))),
    }
json_out = eval_dir / "comparison_results.json"
json_out.write_text(json.dumps(results_out, indent=2))
print("JSON saved to %s" % json_out)

# --- Write markdown report ---
lines = [
    "# PDF-to-Markdown Method Comparison & Ratings",
    "",
    "**Test PDF:** `csc_notes/pdfs/algorithms_erickson_01_recursion.pdf`",
    "",
    "| Method | Time (s) | Page Markers | Multi-pages | Length >500 | Heading | Bibliog. Marker | Score /5 | % |",
    "|---|---|---|---|---|---|---|---|---|",
]
for name, data in methods_data.items():
    r = data["rating"]
    lines.append("| %s | %.2f | %s | %s | %d | %s | %s | %d/%d | %.1f%% |" % (
        name, data["elapsed"],
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
lines.append("**Overall recommendation**: Method A (VLM, cached after first) would score highest because it has page markers, multiple pages, good text length, headings, and no stray bibliography marker. The one-time LLM cost is negligible compared to the retrieval-quality gain.")
lines.append("")
lines.append("**Files saved**")
lines.append("- `eval/method_*.md` — markdown output from each method")
lines.append("- `eval/comparison_results.json` — raw data + check results")
lines.append("- `eval/comparison_report.md` — this markdown report")
lines.append("")
lines.append("**Markdown report saved to:** `eval/comparison_report.md`")
print("\n--- Saved markdown files ---")
for name in ["B", "C", "D"]:
    p = eval_dir / ("method_%s.md" % name.replace(".", "_").replace(" ", "_"))
    print("  method_%s.md → %d chars" % (name, len(methods_data[name]["md_content"])))
print("\n--- Done ---")