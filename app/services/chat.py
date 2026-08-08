from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chat


class ChatService:
    async def create_chat(self, user_id: int, title: str, db: AsyncSession) -> Chat:
        chat = Chat(user_id=user_id, title=title)
        db.add(chat)
        await db.commit()
        await db.refresh(chat)
        return chat

    async def list_chats(self, user_id: int, db: AsyncSession) -> list[Chat]:
        result = await db.execute(
            select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
        )
        return list(result.scalars().all())

    async def get_chat(self, chat_id: int, user_id: int, db: AsyncSession) -> Chat:
        result = await db.execute(select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id))
        chat = result.scalar_one_or_none()
        if chat is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat not found")
        return chat

    async def delete_chat(self, chat_id: int, user_id: int, db: AsyncSession) -> None:
        chat = await self.get_chat(chat_id, user_id, db)
        await db.delete(chat)
        await db.commit()


chat_service = ChatService()
