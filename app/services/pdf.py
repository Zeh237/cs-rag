import base64
import hashlib
import io
import json
import os
import re
import time

import fitz
from fastapi.responses import FileResponse, RedirectResponse, Response

from app.core.config import PROJECT_ROOT, settings
from app.services.elasticsearch import es_service

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class PdfService:
    def __init__(self) -> None:
        self.notes_manifest_path = settings.NOTES_MANIFEST
        self.index_name = settings.INDEX_NAME
        self._cache: dict[str, tuple[float, bytes]] = {}
        self._cache_ttl = 600
        self._cache_max = 20

    def _get_cached(self, key: str) -> bytes | None:
        entry = self._cache.get(key)
        if entry and time.time() - entry[0] < self._cache_ttl:
            return entry[1]
        self._cache.pop(key, None)
        return None

    def _set_cached(self, key: str, data: bytes) -> None:
        self._cache[key] = (time.time(), data)
        if len(self._cache) > self._cache_max:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)

    def serve(self, paper_id: str, chunk_id: str | None = None, range_header: str | None = None) -> Response:
        notes_entry = self._find_notes_entry(paper_id)
        if notes_entry is None:
            raise FileNotFoundError(f"Note '{paper_id}' not found in the notes manifest.")

        local_path = self.local_pdf_path(paper_id)
        if local_path:
            return self._serve_note_pdf(local_path, chunk_id, range_header)
        if notes_entry.get("url"):
            return RedirectResponse(notes_entry["url"], status_code=302)
        raise FileNotFoundError(f"Note '{paper_id}' has no local PDF or source URL.")

    def _plain_response(self, local_path: str) -> Response:
        return FileResponse(local_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})

    def _serve_note_pdf(self, pdf_path: str, chunk_id: str | None = None, range_header: str | None = None) -> Response:
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(PROJECT_ROOT, pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF file not found on disk.")
        if not chunk_id:
            return self._plain_response(pdf_path)

        cache_key = f"note:{pdf_path}:{chunk_id}"
        data = self._get_cached(cache_key)
        if data is None:
            try:
                data = self._build_highlighted_pdf(pdf_path, chunk_id)
            except Exception:
                data = None
            if data is None:
                data = self._raw_pdf_bytes(pdf_path)
            self._set_cached(cache_key, data)
        return self._serve_bytes(data, range_header)

    def _build_highlighted_pdf(self, pdf_path: str, chunk_id: str) -> bytes | None:
        """Return a re-serialized PDF with highlights, or None if no highlight
        could be placed (then the pristine file is served instead).

        The scan window comes from the chunk's own page span (plus a small
        margin) so the fallback never sweeps a whole multi-hundred-page book,
        which was the source of the multi-second first load.
        """
        try:
            source = es_service.client.get(index=self.index_name, id=chunk_id)["_source"]
        except Exception:
            return None

        doc = fitz.open(pdf_path)
        try:
            page_start = int(source.get("page_start") or source.get("page_number") or 1)
            page_end = int(source.get("page_end") or page_start)
            matches = self._highlight_by_search(doc, source["text"], range(max(0, page_start - 2), min(len(doc), page_end + 2)))
            if matches == 0:
                matches = self._highlight_by_search(doc, source["text"], range(max(0, page_start - 6), min(len(doc), page_end + 6)))
            if matches == 0:
                return None

            buffer = io.BytesIO()
            doc.save(buffer)
            return buffer.getvalue()
        finally:
            doc.close()

    def _raw_pdf_bytes(self, pdf_path: str) -> bytes:
        cache_key = f"raw:{pdf_path}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        with open(pdf_path, "rb") as f:
            data = f.read()
        self._set_cached(cache_key, data)
        return data

    def _serve_bytes(self, data: bytes, range_header: str | None) -> Response:
        headers = {
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, max-age=300",
            "ETag": f'"{hashlib.md5(data).hexdigest()}"',
        }
        if range_header:
            match = _RANGE_RE.match(range_header.strip())
            if match:
                total = len(data)
                start_s, end_s = match.group(1), match.group(2)
                if start_s == "":
                    suffix = int(end_s) if end_s else 0
                    start, end = max(0, total - suffix), total - 1
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s else total - 1
                start = min(start, total)
                end = min(end, total - 1)
                if start > end or start >= total:
                    return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})
                headers["Content-Range"] = f"bytes {start}-{end}/{total}"
                return Response(content=data[start : end + 1], status_code=206, media_type="application/pdf", headers=headers)
        return Response(content=data, media_type="application/pdf", headers=headers)

    def _notes_pdf_path(self, paper_id: str) -> str | None:
        """Find a local PDF for a note even when the manifest lacks pdf_path."""
        notes_dir = os.path.join(PROJECT_ROOT, settings.NOTES_DIR)
        for candidate in (
            os.path.join(notes_dir, "pdfs", f"{paper_id}.pdf"),
            os.path.join(notes_dir, f"{paper_id}.pdf"),
        ):
            if os.path.isfile(candidate):
                return candidate
        return None

    def local_pdf_path(self, paper_id: str) -> str | None:
        """Path to the local PDF for a note, or None."""
        notes_entry = self._find_notes_entry(paper_id)
        if notes_entry is None:
            return None
        candidate = notes_entry.get("pdf_path") or self._notes_pdf_path(paper_id)
        if not candidate:
            return None
        base_path = os.path.join(PROJECT_ROOT, candidate) if not os.path.isabs(candidate) else candidate
        return base_path if os.path.isfile(base_path) else None

    def has_local_pdf(self, paper_id: str) -> bool:
        return self.local_pdf_path(paper_id) is not None

    def render_page_images(self, paper_id: str, pages: list[int], dpi: int = 110) -> list[dict]:
        """Render PDF pages of a note as base64 PNG data-URIs for a vision model.

        Pages are 1-based and clamped to the document length. If a page cannot be
        rasterized, its extracted text is returned instead so the caller can fall
        back to text. Returns a list of {"page", "image"|None, "text"}."""
        local_path = self.local_pdf_path(paper_id)
        if not local_path:
            return []
        cache_key = f"{paper_id}:{tuple(sorted(set(pages)))}:{dpi}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        doc = fitz.open(local_path)
        try:
            n_pages = doc.page_count
            page_nums = sorted({max(1, min(int(raw), n_pages)) for raw in pages})
            results = []
            for page_num in page_nums:
                page = doc[page_num - 1]
                entry = {"page": page_num, "image": None, "text": page.get_text("text").strip()}
                try:
                    pix = page.get_pixmap(dpi=dpi)
                    entry["image"] = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
                except Exception:
                    pass
                results.append(entry)
        finally:
            doc.close()
        self._set_cached(cache_key, results)
        return results

    def _find_notes_entry(self, paper_id: str) -> dict | None:
        if not os.path.exists(self.notes_manifest_path):
            return None
        try:
            with open(self.notes_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            return None
        return self._match_entry(manifest, paper_id)

    def _match_entry(self, manifest: list[dict], paper_id: str) -> dict | None:
        return next(
            (
                item
                for item in manifest
                if item.get("id") == paper_id
                or os.path.splitext(os.path.basename(item.get("local_path", "")))[0] == paper_id
            ),
            None,
        )

    def _highlight_by_search(self, doc: fitz.Document, chunk_text: str, page_range: range) -> int:
        cleaned = re.sub(r"^(#.*\n|Source:.*\n)+", "", chunk_text)
        words = cleaned.split()
        phrases = []
        phrase_len = 12
        step = 8
        for start in range(0, len(words), step):
            phrase = " ".join(words[start : start + phrase_len])
            if len(phrase) > 5:
                phrases.append(phrase)

        matches = 0
        for pg_idx in page_range:
            page = doc[pg_idx]
            for phrase in phrases:
                for rect in page.search_for(phrase):
                    annot = page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1, 0.8, 0))
                    annot.update()
                    matches += 1
        return matches

pdf_service = PdfService()
