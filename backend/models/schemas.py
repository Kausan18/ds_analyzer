from pydantic import BaseModel
from typing import List, Optional


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    session_id: str
    question: str
    history: Optional[List[ChatMessage]] = []


class ReembedRequest(BaseModel):
    session_id: str
    report: dict