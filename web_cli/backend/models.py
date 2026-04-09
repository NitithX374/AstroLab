from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Any
from datetime import datetime

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

class AskRequest(BaseModel):
    prompt: str

class MessageDB(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    timestamp: datetime

    model_config = ConfigDict(populate_by_name=True)

class ConversationDB(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(populate_by_name=True)

class AskLog(BaseModel):
    id: str
    user_id: str
    prompt: str
    response: str
    tokens_used: int
    latency_ms: float
    model: str
    status: str
    created_at: datetime
