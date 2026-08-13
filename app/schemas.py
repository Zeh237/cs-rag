from pydantic import BaseModel, Field

from app.db.models import Chat


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    username: str

    model_config = {"from_attributes": True}


class CreateChatRequest(BaseModel):
    title: str = "New Chat"


class ChatResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str

    @classmethod
    def from_chat(cls, chat: Chat) -> "ChatResponse":
        return cls(
            id=chat.id,
            title=chat.title,
            created_at=chat.created_at.isoformat(),
            updated_at=chat.updated_at.isoformat(),
        )


class ChatDetailResponse(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    history: list[dict]


class IngestResponse(BaseModel):
    status: str
    message: str
    paper_id: str
    chunks_indexed: int


class SearchHit(BaseModel):
    score: float
    paper_id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    category: str | None = None
    chunk_id: str | None = None
    page_number: int = 1
    text: str | None = None
    source_url: str | None = None
    local_pdf: bool = False


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class SourceHit(BaseModel):
    paper_id: str | None = None
    title: str | None = None
    anchor_chunk_id: str | None = None
    page_number: int = 1
    page_start: int = 1
    page_end: int = 1
    matching_score: float = 0.0
    source_url: str | None = None
    local_pdf: bool = False


class RagResponse(BaseModel):
    question: str
    answer: str
    search_query: str | None = None
    sources: list[SourceHit] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)
