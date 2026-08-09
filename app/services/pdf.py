import io
import json
import os
import re
import time

import fitz
from fastapi.responses import FileResponse, RedirectResponse, Response, StreamingResponse

from app.core.config import PROJECT_ROOT, settings
from app.services.elasticsearch import es_service


class PdfService:
    def __init__(self) -> None:
        self.manifest_path = settings.MANIFEST_PATH
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

    def serve(self, paper_id: str, chunk_id: str | None = None) -> Response:
        notes_entry = self._find_notes_entry(paper_id)
        if notes_entry:
            local_path = self.local_pdf_path(paper_id)
            if local_path:
                return self._serve_note_pdf(local_path, chunk_id)
            if notes_entry.get("url"):
                return RedirectResponse(notes_entry["url"], status_code=302)

        local_path = self._resolve_path(paper_id)
        if not chunk_id:
            return self._plain_response(local_path)

        cache_key = f"paper:{local_path}:{chunk_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return self._stream_pdf(cached)

        try:
            source = es_service.client.get(index=self.index_name, id=chunk_id)["_source"]
            page_num = source.get("page_number", 1)
            highlight_boxes = source.get("highlight_boxes") or {}

            doc = fitz.open(local_path)
            if highlight_boxes:
                for page_str, rects in highlight_boxes.items():
                    page_idx = int(page_str) - 1
                    if 0 <= page_idx < len(doc):
                        page = doc[page_idx]
                        for rect in rects:
                            annot = page.add_highlight_annot(fitz.Rect(*rect))
                            annot.set_colors(stroke=(1, 0.8, 0))
                            annot.update()
            else:
                self._highlight_by_search(doc, source["text"], page_num)

            buffer = io.BytesIO()
            doc.save(buffer)
            doc.close()
            buffer.seek(0)
            data = buffer.getvalue()
            self._set_cached(cache_key, data)
            return self._stream_pdf(data)
        except Exception:
            return self._plain_response(local_path)

    def _plain_response(self, local_path: str) -> Response:
        return FileResponse(local_path, media_type="application/pdf", headers={"Content-Disposition": "inline"})

    def _serve_note_pdf(self, pdf_path: str, chunk_id: str | None = None) -> Response:
        if not os.path.isabs(pdf_path):
            pdf_path = os.path.join(PROJECT_ROOT, pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("PDF file not found on disk.")
        if not chunk_id:
            return self._plain_response(pdf_path)

        cache_key = f"note:{pdf_path}:{chunk_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return self._stream_pdf(cached)

        try:
            source = es_service.client.get(index=self.index_name, id=chunk_id)["_source"]
            doc = fitz.open(pdf_path)
            page_num = source.get("page_number", 1)
            if not self._highlight_by_search(doc, source["text"], page_num, all_pages=False):
                self._highlight_by_search(doc, source["text"], page_num=1, all_pages=True)

            buffer = io.BytesIO()
            doc.save(buffer)
            doc.close()
            buffer.seek(0)
            data = buffer.getvalue()
            self._set_cached(cache_key, data)
            return self._stream_pdf(data)
        except Exception:
            return self._plain_response(pdf_path)

    def _stream_pdf(self, data: bytes) -> Response:
        return StreamingResponse(io.BytesIO(data), media_type="application/pdf", headers={"Content-Disposition": "inline"})

    def _resolve_path(self, paper_id: str) -> str:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError("Library manifest not found.")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        entry = self._match_entry(manifest, paper_id)
        if entry is None:
            raise FileNotFoundError(f"Paper '{paper_id}' not found in manifest.")

        local_path = entry.get("local_path", "")
        if not os.path.isabs(local_path):
            local_path = os.path.join(PROJECT_ROOT, local_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError("PDF file not found on disk.")
        return local_path

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
        """Path to the PDF we can serve locally for a paper, or None."""
        notes_entry = self._find_notes_entry(paper_id)
        if notes_entry is not None:
            candidate = notes_entry.get("pdf_path") or self._notes_pdf_path(paper_id)
            if not candidate:
                return None
            base_path = os.path.join(PROJECT_ROOT, candidate) if not os.path.isabs(candidate) else candidate
            return base_path if os.path.isfile(base_path) else None

        try:
            candidate = self._resolve_path(paper_id)
        except Exception:
            return None
        return candidate if os.path.isfile(candidate) else None

    def has_local_pdf(self, paper_id: str) -> bool:
        return self.local_pdf_path(paper_id) is not None

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

    def _highlight_by_search(self, doc: fitz.Document, chunk_text: str, page_num: int, all_pages: bool = False) -> int:
        cleaned = re.sub(r"^(#.*\n|Source:.*\n)+", "", chunk_text)
        words = cleaned.split()
        phrases = []
        phrase_len = 12
        step = 8
        for start in range(0, len(words), step):
            phrase = " ".join(words[start : start + phrase_len])
            if len(phrase) > 5:
                phrases.append(phrase)

        if all_pages:
            page_range = range(len(doc))
        else:
            page_range = range(max(0, page_num - 1), min(len(doc), page_num + 1))

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
