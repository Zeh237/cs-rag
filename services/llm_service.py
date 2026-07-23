import httpx
from config import settings
from typing import AsyncGenerator

class LLMService:
    def __init__(self):
        self.api_url = getattr(settings, "LLM_API_BASE") + "/chat/completions"
        self.model_name = getattr(settings, "LLM_MODEL")

    def _build_payload(self, question: str, context_chunks: list[dict], stream: bool = False) -> dict:
        """Builds the request payload for the LLM API."""
        context_text = "\n\n".join([
            f"Source: {chunk['title']} (Chunk: {chunk['chunk_id']})\nContent: {chunk['text']}"
            for chunk in context_chunks
        ])

        system_prompt = (
            "You are an expert computer science research assistant. Your task is to answer the user's query "
            "by synthesizing the provided academic context.\n\n"
            "Guidelines:\n"
            "- Ground your explanation in the provided context as much as possible.\n"
            "- If the user asks a broad, high-level conceptual question (e.g., 'What is X?') and the context "
            "focuses on specific implementations or papers (e.g., Paxos, Raft, state machines), use these "
            "retrieved papers as concrete case studies and examples to explain how the broader concept works in practice.\n"
            "- If the text lacks a basic definition of a high-level term, you may briefly define it using your general "
            "knowledge, but immediately pivot to using the retrieved text to illustrate its practical mechanics, "
            "such as replication, fault tolerance, and consensus mechanisms.\n"
            "- Remain academic, precise, and objective. Avoid making up facts not supported by either the text or standard theory.\n"
            "- If the context is insufficient to answer the question, clearly state that and avoid speculation.\n"
        )

        user_prompt = f"""Use the following academic literature context to address the user's inquiry.

            ---
            CONTEXT:
            {context_text}
            ---

            QUESTION: {question}

            ANSWER:"""

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "stream": stream,
        }

    async def generate_response(self, question: str, context_chunks: list[dict]) -> str:
        """
        Non-streaming: formats retrieved chunks into a prompt and calls the local LM Studio instance.
        Returns the complete answer as a string.
        """
        payload = self._build_payload(question, context_chunks, stream=False)

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                return f"Failed to generate answer from local LLM: {str(e)}"

    async def stream_response(self, question: str, context_chunks: list[dict]) -> AsyncGenerator[str, None]:
        """
        Streaming: yields raw text delta strings as they arrive from the LLM.
        Uses the OpenAI-compatible SSE streaming format supported by LM Studio.
        """
        payload = self._build_payload(question, context_chunks, stream=True)

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                async with client.stream("POST", self.api_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue
            except Exception as e:
                yield f"\n\n[Error communicating with LLM: {str(e)}]"

llm_service = LLMService()