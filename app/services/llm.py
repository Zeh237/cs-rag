import json
from typing import AsyncGenerator, Optional

import httpx

from app.core.config import settings


class LLMService:
    def __init__(self) -> None:
        self.api_url = settings.LLM_API_BASE.rstrip("/") + "/chat/completions"
        self.model_name = settings.LLM_MODEL

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
        context_text = "\n\n".join(
            f"Source: {chunk['title']} (Chunk: {chunk['chunk_id']})\nContent: {chunk['text']}"
            for chunk in context_chunks
        )
        system_prompt = (
            "You are a friendly undergraduate computer science study assistant. Answer the user's question by "
            "synthesizing the provided course notes and past conversation history.\n\n"
            "Guidelines:\n"
            "- Ground your explanation in the provided notes as much as possible; cite the source titles you use.\n"
            "- Define key terms and explain concepts clearly and simply, as if tutoring a student.\n"
            "- Use concrete examples and small walk-throughs from the notes to illustrate ideas.\n"
            "- Keep answers structured (short paragraphs or bullets) and study-oriented.\n"
            "- If the notes are insufficient, say so clearly and avoid inventing facts.\n"
        )
        user_prompt = (
            "Use the following course notes as context to address the user's inquiry.\n\n"
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


llm_service = LLMService()
