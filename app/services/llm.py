import json
from typing import AsyncGenerator, Optional

import httpx

from app.core.config import settings


_RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Search the document for pages that discuss a topic. Returns pages with short excerpts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic or phrase to look for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_pages",
            "description": "Open PDF pages as images so you can read their content, including figures and tables.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "description": "First 1-based page number"},
                    "end": {"type": "integer", "description": "Last 1-based page number, inclusive"},
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_toc",
            "description": "Return the table of contents (headings with page numbers) of the document.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class LLMService:
    def __init__(self) -> None:
        self.api_url = settings.LLM_API_BASE.rstrip("/") + "/chat/completions"
        self.model_name = settings.LLM_MODEL
        self.vision_model_name = settings.VISION_LLM_MODEL or self.model_name

    async def transcribe_page_image(self, image_data_uri: str) -> str:
        prompt = (
            "You are transcribing a single page from computer science course notes (algorithms, ML, OS, etc.). "
            "Extract all substantive content: text, equations, algorithms, tables, figure descriptions, and code snippets. "
            "Maintain structure (headings, bullet points, numbered lists) where possible. "
            "\n\n"
            "IMPORTANT — SKIP these pages entirely (return empty string): "
            "Table of Contents, Index, Bibliography, References, Works Cited, "
            "blank pages, cover pages, dedication pages. "
            "Also ignore running headers/footers (page numbers, chapter titles repeated at top/bottom). "
            "Do not include any introductory or conversational text. "
            "If the page is substantive, transcribe/describe it directly. If it should be skipped, output exactly: [SKIP]"
            "if a page has substantive content alongside anything that should be skipped, transcribe only the substantive content and ignore the rest."
        )
        payload = {
            "model": self.vision_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(self.api_url, json=payload)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"].strip()

    async def rewrite_query_for_search(self, question: str, history: Optional[list[dict[str, str]]] = None) -> str:
        if history:
            summary = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-4:])
            prompt = (
                "Given the following conversation history and a follow-up question, rephrase the follow-up question "
                "to be a standalone search query for a library of undergraduate computer science course notes. "
                "Do NOT answer the question. Return ONLY the rewritten query.\n\n"
                f"Chat History:\n{summary}\n\n"
                f"Follow-up Question: {question}\n"
                "Standalone Search Query:"
            )
        else:
            prompt = (
                "You are converting a user question into a search query for a library of undergraduate computer "
                "science course notes (algorithms, data structures, operating systems, networking, databases, "
                "compilers, theory, discrete math, and more). Extract the core topics and concepts, expand "
                "acronyms and abbreviations where helpful, and drop conversational filler. Do NOT answer the "
                "question. Return ONLY the rewritten search query.\n\n"
                f"Question: {question}\n"
                "Search Query:"
            )

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.api_url, json=payload)
                if response.status_code == 200:
                    rewritten = response.json()["choices"][0]["message"]["content"].strip()
                    if rewritten:
                        return rewritten
        except Exception:
            pass
        return question

    def _build_payload(self, question: str, context_chunks: list[dict], history=None, stream: bool = False) -> dict:
        context_lines = []
        for idx, chunk in enumerate(context_chunks, start=1):
            title = chunk.get("title") or "Course notes"
            page_start = chunk.get("page_start") or chunk.get("page_number") or 1
            page_end = chunk.get("page_end") or page_start
            pages = f"p. {page_start}" if page_start == page_end else f"pp. {page_start}-{page_end}"
            context_lines.append(f"[{idx}] {title} ({pages}):\n{chunk.get('text', '')}")
        context_text = "\n\n".join(context_lines)

        system_prompt = (
            "You are a friendly undergraduate computer science study assistant. Answer the user's question by "
            "synthesizing the provided course notes and past conversation history.\n\n"
            "Guidelines:\n"
            "- Ground every claim in the provided notes; cite your sources inline with bracketed numbers that "
            "match the numbered context list, e.g. [1] or [2][3].\n"
            "- Each source header shows the exact page range the text came from. When you reference specific "
            "pages, give them as (p. 40) or (pp. 40-43) and ONLY use pages that appear in the source headers; "
            "never guess or invent page numbers.\n"
            "- Define key terms and explain concepts clearly and simply, as if tutoring a student.\n"
            "- Use concrete examples and small walk-throughs from the notes to illustrate ideas.\n"
            "- Keep answers structured (short paragraphs or bullets) and study-oriented.\n"
            "- If the notes are insufficient, say so clearly and avoid inventing facts.\n"
        )
        user_prompt = (
            "Use the following course notes as context to address the user's inquiry. Each source is labeled "
            "with its number and page range.\n\n"
            f"---\nCONTEXT:\n{context_text}\n---\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return {"model": self.model_name, "messages": messages, "temperature": 0.3, "stream": stream}

    async def generate_response(self, question: str, context_chunks: list[dict], history=None) -> str:
        payload = self._build_payload(question, context_chunks, history=history, stream=False)
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Failed to generate answer from local LLM: {e}"

    async def stream_response(self, question: str, context_chunks: list[dict], history=None) -> AsyncGenerator[str, None]:
        payload = self._build_payload(question, context_chunks, history=history, stream=True)
        async with httpx.AsyncClient(timeout=300.0) as client:
            try:
                async with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            break
                        try:
                            delta = json.loads(data)["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
            except Exception as e:
                yield f"\n\n[Error communicating with LLM: {e}]"

    async def stream_research_response(
        self,
        question: str,
        *,
        paper_id: str,
        title: str,
        seed_images: list[dict],
        callbacks: dict[str, callable],
        toc_hint: str = "",
        history: Optional[list[dict[str, str]]] = None,
        max_pages: int = 12,
        max_tool_iters: int = 8,
    ) -> AsyncGenerator[str, None]:
        """Vision research mode: streams the model's final answer after it has
        used tools to read pages of a PDF. Tool results are returned as text
        (tool-role) with page images delivered in follow-up user messages,
        matching LM Studio's multimodal tool support."""
        system = (
            "You are a study assistant that researches a single PDF document to answer the user's question. "
            "The document's pages are shown to you as images; use the tools to read more pages or locate topics.\n"
            "Rules:\n"
            "- Call read_pages to open the pages you need; the images are the actual document pages.\n"
            "- Call search_document to find which pages discuss a topic.\n"
            "- Cite the page of every claim inline as (p. N) or (pp. N-M). ONLY cite pages you actually saw "
            "or that search_document returned; never invent page numbers.\n"
            "- If the document does not cover the question, say so clearly.\n"
            "- Answer in structured, study-friendly prose.\n"
        )
        user_text = f"QUESTION: {question}\n\nDOCUMENT: {title}"
        if toc_hint:
            user_text += f"\n\nTABLE OF CONTENTS:\n{toc_hint}"
        user_text += "\n\nSeed pages are shown below; read more pages if you need them."

        content: list[dict] = [{"type": "text", "text": user_text}]
        for img in seed_images:
            if img.get("image"):
                content.append({"type": "text", "text": f"Page {img['page']}:"})
                content.append({"type": "image_url", "image_url": {"url": img["image"]}})
            elif img.get("text"):
                content.append({"type": "text", "text": f"Page {img['page']} text: {img['text'][:600]}"})
        if not any(part.get("type") == "image_url" for part in content):
            return  # nothing readable

        messages: list[dict] = [{"role": "system", "content": system}]
        if history:
            messages.extend(
                [m for m in history if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
            )
        messages.append({"role": "user", "content": content})

        pages_read = {img.get("page") for img in seed_images if img.get("image")}
        limit_notified = False

        for _ in range(max_tool_iters):
            payload = {
                "model": self.vision_model_name,
                "messages": messages,
                "tools": _RESEARCH_TOOLS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "stream": True,
            }
            acc: dict[int, dict] = {}
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    async with client.stream("POST", self.api_url, json=payload) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:") :].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except Exception:
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta", {})
                            token = delta.get("content")
                            if token:
                                yield token
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                entry = acc.setdefault(idx, {"id": tc.get("id") or "", "name": "", "arguments": ""})
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    entry["name"] += fn["name"]
                                if fn.get("arguments"):
                                    entry["arguments"] += fn["arguments"]
                            if choice.get("finish_reason") == "tool_calls":
                                break
            except Exception as e:
                yield f"\n\n[Error communicating with vision LLM: {e}]"
                return

            if not acc:
                return  # finish_reason stop: final text already streamed

            assistant_msg: dict = {"role": "assistant", "content": None, "tool_calls": []}
            for idx in sorted(acc):
                entry = acc[idx]
                tc_id = entry["id"] or f"call_{len(messages)}"
                assistant_msg["tool_calls"].append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": entry["name"], "arguments": entry["arguments"]},
                    }
                )
            messages.append(assistant_msg)

            for idx in sorted(acc):
                entry = acc[idx]
                name = entry["name"]
                try:
                    args = json.loads(entry["arguments"] or "{}")
                except Exception:
                    args = {}
                tc_id = entry["id"] or f"call_{len(messages)}"

                if name == "read_pages":
                    try:
                        start = max(1, int(args.get("start", 1)))
                        end = max(1, int(args.get("end", start)))
                    except Exception:
                        start, end = 1, 1
                    if end < start:
                        start, end = end, start
                    wanted = list(range(start, end + 1))
                    new_pages = [p for p in wanted if p not in pages_read]
                    if wanted and not new_pages:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": f"Pages {start}-{end} were already shown above; continue using them.",
                            }
                        )
                        continue
                    rendered = await callbacks["read_pages"](new_pages)
                    pages_read.update(r.get("page") for r in rendered)
                    parts: list[dict] = []
                    for r in rendered:
                        if r.get("image"):
                            parts.append({"type": "text", "text": f"Page {r['page']}:"})
                            parts.append({"type": "image_url", "image_url": {"url": r["image"]}})
                        elif r.get("text"):
                            parts.append({"type": "text", "text": f"Page {r['page']} text: {r['text'][:500]}"})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": f"Rendering pages {start}-{end} as images below."
                            if rendered
                            else f"Could not render pages {start}-{end}; use search_document instead.",
                        }
                    )
                    if parts:
                        messages.append({"role": "user", "content": parts})
                    if not limit_notified and len(pages_read) >= max_pages:
                        limit_notified = True
                        messages.append(
                            {
                                "role": "user",
                                "content": f"You have reached the limit of {max_pages} pages for this query. "
                                "Answer now using what you have read, citing pages with (p. N).",
                            }
                        )
                elif name == "search_document":
                    query = args.get("query") or question
                    result_text = await callbacks["search_document"](query)
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
                elif name == "get_toc":
                    result_text = await callbacks["get_toc"]()
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
                else:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": "Unknown tool."})

        yield "\n\n[Reached the tool-use limit; answer may be truncated.]"


llm_service = LLMService()
