from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ReembedRequest(BaseModel):
    session_id: str
    report: dict