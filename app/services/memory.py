import json

import redis.asyncio as redis

from app.core.config import settings


class MemoryStore:
    def __init__(self, ttl: int = 86400, max_turns: int = 6) -> None:
        self.ttl = ttl
        self.max_messages = max_turns * 2
        kwargs = {"decode_responses": True}
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        self.client = redis.from_url(settings.REDIS_HOST, **kwargs)

    def _get_key(self, session_id: str) -> str:
        return f"chat:session:{session_id}"

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        raw_messages = await self.client.lrange(self._get_key(session_id), 0, -1)
        return [json.loads(msg) for msg in raw_messages]

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        key = self._get_key(session_id)
        payload = json.dumps({"role": role, "content": content})
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl)
            await pipe.execute()

    async def clear_session(self, session_id: str) -> None:
        await self.client.delete(self._get_key(session_id))

    async def close(self) -> None:
        await self.client.aclose()


memory_store = MemoryStore()
