from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class Speaker(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    is_customer: bool = False
    title: Optional[str] = None

class TranscriptEntry(BaseModel):
    speaker: Speaker
    start_time: float
    end_time: float
    text: str

class CallMetadata(BaseModel):
    call_id: str
    title: str
    date: datetime
    duration_seconds: int
    participants: list[Speaker]
    customer_company: Optional[str] = None
    industry: Optional[str] = None
    deal_stage: Optional[str] = None
    call_type: Optional[str] = None
    url: Optional[str] = None

class GongCall(BaseModel):
    metadata: CallMetadata
    transcript: list[TranscriptEntry]

class ChunkMetadata(BaseModel):
    call_id: str
    call_title: str
    call_date: str
    chunk_index: int
    total_chunks: int
    speakers_in_chunk: list[str]
    customer_company: Optional[str] = None
    industry: Optional[str] = None
    deal_stage: Optional[str] = None
    start_time: float
    end_time: float
    has_customer_speech: bool = False
    has_rep_speech: bool = False

class TranscriptChunk(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[ChatMessage] = []
    filters: Optional[dict] = None

class SourceReference(BaseModel):
    call_id: str
    call_title: str
    call_date: str
    customer_company: Optional[str]
    relevance_score: float
    excerpt: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    query_time_ms: int

class SyncStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class SyncJob(BaseModel):
    job_id: str
    status: SyncStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    calls_processed: int = 0
    chunks_created: int = 0
    errors: list[str] = []

class SyncRequest(BaseModel):
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    call_ids: Optional[list[str]] = None

class IndexStats(BaseModel):
    total_vectors: int
    total_calls: int
    date_range: dict
    top_companies: list[dict]
    top_industries: list[dict]
