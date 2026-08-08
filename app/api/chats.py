from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import get_db
from app.schemas import ChatDetailResponse, ChatResponse, CreateChatRequest
from app.services.auth import auth_service
from app.services.chat import chat_service
from app.services.memory import memory_store

router = APIRouter(prefix="/chats", tags=["Chats"])


@router.post("", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create(
    body: CreateChatRequest,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await chat_service.create_chat(current_user.id, body.title, db)
    return ChatResponse.from_chat(chat)


@router.get("", response_model=list[ChatResponse])
async def list_all(
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chats = await chat_service.list_chats(current_user.id, db)
    return [ChatResponse.from_chat(chat) for chat in chats]


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_one(
    chat_id: int,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    chat = await chat_service.get_chat(chat_id, current_user.id, db)
    return ChatDetailResponse(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at.isoformat(),
        updated_at=chat.updated_at.isoformat(),
        history=await memory_store.get_history(str(chat_id)),
    )


@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove(
    chat_id: int,
    current_user: User = Depends(auth_service.get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await chat_service.get_chat(chat_id, current_user.id, db)
    await memory_store.clear_session(str(chat_id))
    await chat_service.delete_chat(chat_id, current_user.id, db)
