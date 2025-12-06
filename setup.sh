#!/bin/bash

# Gong RAG Chatbot - Project Setup Script
# Run: chmod +x setup.sh && ./setup.sh

set -e

echo "🚀 Creating Gong RAG Chatbot project structure..."

# Create directories
mkdir -p gong-rag-chatbot/{backend/app,frontend/src/components}
cd gong-rag-chatbot

# ============================================
# BACKEND FILES
# ============================================

# requirements.txt
cat > backend/requirements.txt << 'EOF'
# FastAPI and server
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6

# Gong API
requests==2.31.0
httpx==0.26.0

# Vector store and embeddings
pinecone-client==3.0.0
openai==1.12.0
tiktoken==0.6.0

# LangChain for RAG
langchain==0.1.6
langchain-openai==0.0.5
langchain-pinecone==0.0.2
langchain-community==0.0.19

# Text processing
nltk==3.8.1

# Configuration and utilities
python-dotenv==1.0.1
pydantic==2.6.0
pydantic-settings==2.1.0

# Async support
aiohttp==3.9.3

# Logging and monitoring
structlog==24.1.0

# Testing
pytest==8.0.0
pytest-asyncio==0.23.4
EOF

# .env.example
cat > backend/.env.example << 'EOF'
# Gong API
GONG_ACCESS_KEY=your_gong_access_key
GONG_ACCESS_KEY_SECRET=your_gong_secret

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=gong-transcripts

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# App Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
EOF

# backend/app/__init__.py
touch backend/app/__init__.py

# config.py
cat > backend/app/config.py << 'EOF'
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    gong_access_key: str
    gong_access_key_secret: str
    gong_base_url: str = "https://api.gong.io/v2"
    pinecone_api_key: str
    pinecone_index_name: str = "gong-transcripts"
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    max_context_tokens: int = 8000
    app_name: str = "Gong RAG Chatbot"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
EOF

# models.py
cat > backend/app/models.py << 'EOF'
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
EOF

# gong_client.py
cat > backend/app/gong_client.py << 'EOF'
import httpx
import base64
from datetime import datetime
from typing import Optional, AsyncIterator
import structlog
from app.config import get_settings
from app.models import GongCall, CallMetadata, TranscriptEntry, Speaker

log = structlog.get_logger()

class GongClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.gong_base_url
        creds = f"{settings.gong_access_key}:{settings.gong_access_key_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        self.headers = {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            resp = await client.request(method, url, headers=self.headers, timeout=60.0, **kwargs)
            resp.raise_for_status()
            return resp.json()
    
    async def list_calls(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None, cursor: Optional[str] = None) -> tuple[list[dict], Optional[str]]:
        payload = {"filter": {}}
        if from_date: payload["filter"]["fromDateTime"] = from_date.isoformat() + "Z"
        if to_date: payload["filter"]["toDateTime"] = to_date.isoformat() + "Z"
        if cursor: payload["cursor"] = cursor
        data = await self._request("POST", "/calls", json=payload)
        return data.get("calls", []), data.get("records", {}).get("cursor")
    
    async def get_call_transcript(self, call_id: str) -> Optional[list[dict]]:
        try:
            data = await self._request("POST", "/calls/transcript", json={"filter": {"callIds": [call_id]}})
            transcripts = data.get("callTranscripts", [])
            return transcripts[0]["transcript"] if transcripts else None
        except Exception as e:
            log.error("transcript_fetch_failed", call_id=call_id, error=str(e))
            return None
    
    async def get_call_details(self, call_ids: list[str]) -> list[dict]:
        data = await self._request("POST", "/calls/extensive", json={"filter": {"callIds": call_ids}})
        return data.get("calls", [])
    
    def _parse_speaker(self, party: dict, is_customer: bool) -> Speaker:
        return Speaker(id=party.get("speakerId", party.get("id", "unknown")), name=party.get("name", "Unknown"), email=party.get("emailAddress"), is_customer=is_customer, title=party.get("title"))
    
    def _parse_call_metadata(self, call: dict, details: Optional[dict] = None) -> CallMetadata:
        participants, customer_company = [], None
        if details:
            for party in details.get("parties", []):
                is_cust = party.get("affiliation") == "Customer"
                participants.append(self._parse_speaker(party, is_cust))
                if is_cust and party.get("company"): customer_company = party.get("company")
        return CallMetadata(call_id=call["id"], title=call.get("title", "Untitled Call"), date=datetime.fromisoformat(call["started"].replace("Z", "+00:00")), duration_seconds=call.get("duration", 0), participants=participants, customer_company=customer_company, industry=details.get("customData", {}).get("industry") if details else None, deal_stage=details.get("customData", {}).get("dealStage") if details else None, call_type=call.get("purpose"), url=call.get("url"))
    
    def _parse_transcript(self, raw: list[dict], speakers: list[Speaker]) -> list[TranscriptEntry]:
        speaker_map = {s.id: s for s in speakers}
        entries = []
        for seg in raw:
            spk_id = seg.get("speakerId", "unknown")
            spk = speaker_map.get(spk_id, Speaker(id=spk_id, name="Unknown"))
            for sent in seg.get("sentences", []):
                entries.append(TranscriptEntry(speaker=spk, start_time=sent.get("start", 0)/1000, end_time=sent.get("end", 0)/1000, text=sent.get("text", "")))
        return entries
    
    async def fetch_calls_with_transcripts(self, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None, call_ids: Optional[list[str]] = None) -> AsyncIterator[GongCall]:
        if call_ids:
            calls = [{"id": cid} for cid in call_ids]
        else:
            all_calls, cursor = [], None
            while True:
                batch, cursor = await self.list_calls(from_date, to_date, cursor)
                all_calls.extend(batch)
                if not cursor: break
            calls = all_calls
        log.info("processing_calls", total=len(calls))
        for i in range(0, len(calls), 20):
            batch_ids = [c["id"] for c in calls[i:i+20]]
            details_list = await self.get_call_details(batch_ids)
            details_map = {d["metaData"]["id"]: d for d in details_list}
            for call in calls[i:i+20]:
                call_id = call["id"]
                details = details_map.get(call_id)
                transcript_raw = await self.get_call_transcript(call_id)
                if not transcript_raw: continue
                metadata = self._parse_call_metadata(call, details)
                transcript = self._parse_transcript(transcript_raw, metadata.participants)
                yield GongCall(metadata=metadata, transcript=transcript)

_client: Optional[GongClient] = None
def get_gong_client() -> GongClient:
    global _client
    if _client is None: _client = GongClient()
    return _client
EOF

# chunker.py
cat > backend/app/chunker.py << 'EOF'
import tiktoken
import hashlib
import structlog
from app.config import get_settings
from app.models import GongCall, TranscriptChunk, ChunkMetadata, TranscriptEntry

log = structlog.get_logger()

class TranscriptChunker:
    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _format_entry(self, entry: TranscriptEntry) -> str:
        speaker_type = "Customer" if entry.speaker.is_customer else "Rep"
        return f"[{speaker_type} - {entry.speaker.name}]: {entry.text}"
    
    def _generate_chunk_id(self, call_id: str, chunk_index: int, text: str) -> str:
        return hashlib.md5(f"{call_id}:{chunk_index}:{text[:100]}".encode()).hexdigest()
    
    def _create_chunk(self, call: GongCall, entries: list[TranscriptEntry], chunk_index: int, total_chunks: int) -> TranscriptChunk:
        text = "\n".join(self._format_entry(e) for e in entries)
        speakers = list(set(e.speaker.name for e in entries))
        metadata = ChunkMetadata(call_id=call.metadata.call_id, call_title=call.metadata.title, call_date=call.metadata.date.strftime("%Y-%m-%d"), chunk_index=chunk_index, total_chunks=total_chunks, speakers_in_chunk=speakers, customer_company=call.metadata.customer_company, industry=call.metadata.industry, deal_stage=call.metadata.deal_stage, start_time=entries[0].start_time, end_time=entries[-1].end_time, has_customer_speech=any(e.speaker.is_customer for e in entries), has_rep_speech=any(not e.speaker.is_customer for e in entries))
        return TranscriptChunk(id=self._generate_chunk_id(call.metadata.call_id, chunk_index, text), text=text, metadata=metadata)
    
    def chunk_call(self, call: GongCall) -> list[TranscriptChunk]:
        if not call.transcript: return []
        turns, current_turn, current_speaker = [], [], None
        for entry in call.transcript:
            if entry.speaker.id != current_speaker and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_speaker = entry.speaker.id
            current_turn.append(entry)
        if current_turn: turns.append(current_turn)
        
        chunks_data, current_entries, current_tokens = [], [], 0
        for turn in turns:
            turn_text = "\n".join(self._format_entry(e) for e in turn)
            turn_tokens = self._count_tokens(turn_text)
            if current_tokens + turn_tokens > self.chunk_size and current_entries:
                chunks_data.append(current_entries)
                overlap_entries, overlap_tokens = [], 0
                for e in reversed(current_entries):
                    e_tok = self._count_tokens(self._format_entry(e))
                    if overlap_tokens + e_tok <= self.chunk_overlap:
                        overlap_entries.insert(0, e)
                        overlap_tokens += e_tok
                    else: break
                current_entries, current_tokens = overlap_entries, overlap_tokens
            current_entries.extend(turn)
            current_tokens += turn_tokens
        if current_entries: chunks_data.append(current_entries)
        chunks_data = [c for c in chunks_data if self._count_tokens("\n".join(self._format_entry(e) for e in c)) >= self.min_chunk_size]
        chunks = [self._create_chunk(call, entries, idx, len(chunks_data)) for idx, entries in enumerate(chunks_data)]
        log.info("chunked_call", call_id=call.metadata.call_id, chunks_created=len(chunks))
        return chunks

def chunk_transcript(call: GongCall) -> list[TranscriptChunk]:
    return TranscriptChunker().chunk_call(call)
EOF

# pinecone_store.py
cat > backend/app/pinecone_store.py << 'EOF'
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from typing import Optional
import structlog
from app.config import get_settings
from app.models import TranscriptChunk, IndexStats

log = structlog.get_logger()

class PineconeStore:
    def __init__(self):
        settings = get_settings()
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.embedding_model = settings.embedding_model
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)
    
    def _ensure_index_exists(self):
        existing = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            log.info("creating_pinecone_index", name=self.index_name)
            self.pc.create_index(name=self.index_name, dimension=1536, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    
    def _get_embedding(self, text: str) -> list[float]:
        return self.openai.embeddings.create(model=self.embedding_model, input=text).data[0].embedding
    
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [item.embedding for item in self.openai.embeddings.create(model=self.embedding_model, input=texts).data]
    
    def _chunk_to_metadata(self, chunk: TranscriptChunk) -> dict:
        return {"call_id": chunk.metadata.call_id, "call_title": chunk.metadata.call_title, "call_date": chunk.metadata.call_date, "chunk_index": chunk.metadata.chunk_index, "total_chunks": chunk.metadata.total_chunks, "speakers": ",".join(chunk.metadata.speakers_in_chunk), "customer_company": chunk.metadata.customer_company or "", "industry": chunk.metadata.industry or "", "deal_stage": chunk.metadata.deal_stage or "", "start_time": chunk.metadata.start_time, "end_time": chunk.metadata.end_time, "has_customer_speech": chunk.metadata.has_customer_speech, "has_rep_speech": chunk.metadata.has_rep_speech, "text": chunk.text}
    
    async def upsert_chunks(self, chunks: list[TranscriptChunk], batch_size: int = 100):
        if not chunks: return
        log.info("upserting_chunks", count=len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = self._get_embeddings_batch([c.text for c in batch])
            vectors = [{"id": c.id, "values": emb, "metadata": self._chunk_to_metadata(c)} for c, emb in zip(batch, embeddings)]
            self.index.upsert(vectors=vectors)
    
    async def search(self, query: str, top_k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        query_emb = self._get_embedding(query)
        pc_filter = None
        if filters:
            pc_filter = {k: {"$in": v} if isinstance(v, list) else {"$eq": v} for k, v in filters.items()}
        results = self.index.query(vector=query_emb, top_k=top_k, include_metadata=True, filter=pc_filter)
        return [{"id": m.id, "score": m.score, "text": m.metadata.get("text", ""), "call_id": m.metadata.get("call_id"), "call_title": m.metadata.get("call_title"), "call_date": m.metadata.get("call_date"), "customer_company": m.metadata.get("customer_company"), "industry": m.metadata.get("industry"), "chunk_index": m.metadata.get("chunk_index"), "has_customer_speech": m.metadata.get("has_customer_speech")} for m in results.matches]
    
    async def delete_by_call_id(self, call_id: str):
        self.index.delete(filter={"call_id": {"$eq": call_id}})
    
    async def get_stats(self) -> IndexStats:
        stats = self.index.describe_index_stats()
        return IndexStats(total_vectors=stats.total_vector_count, total_calls=0, date_range={}, top_companies=[], top_industries=[])

_store: Optional[PineconeStore] = None
def get_pinecone_store() -> PineconeStore:
    global _store
    if _store is None: _store = PineconeStore()
    return _store
EOF

# rag_pipeline.py
cat > backend/app/rag_pipeline.py << 'EOF'
from openai import OpenAI
from typing import Optional, AsyncIterator
import structlog
import time
from app.config import get_settings
from app.pinecone_store import get_pinecone_store
from app.models import ChatRequest, ChatResponse, ChatMessage, SourceReference

log = structlog.get_logger()

SYSTEM_PROMPT = """You are an AI assistant that analyzes sales and customer call transcripts to help marketing and sales teams gain insights. You have access to a database of Gong call transcripts.

Your role is to:
1. Answer questions about patterns, trends, and insights from customer calls
2. Identify common themes, objections, use cases, and competitor mentions
3. Provide specific examples and quotes from calls when relevant
4. Help identify customer personas, industries, and deal characteristics

When answering:
- Be specific and cite evidence from the call transcripts provided
- Quantify patterns when possible (e.g., "mentioned in 3 out of 5 relevant calls")
- Distinguish between customer statements and sales rep statements
- If the provided context doesn't contain enough information, say so clearly"""

class RAGPipeline:
    def __init__(self):
        settings = get_settings()
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.chat_model = settings.chat_model
        self.store = get_pinecone_store()
        self.retrieval_top_k = settings.retrieval_top_k
        self.rerank_top_k = settings.rerank_top_k
    
    def _enhance_query(self, question: str) -> str:
        resp = self.openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"Generate an enhanced search query for finding relevant sales call transcripts. Question: {question}\n\nReturn only the enhanced query."}], max_tokens=100, temperature=0.3)
        return resp.choices[0].message.content.strip()
    
    def _format_context(self, matches: list[dict]) -> str:
        parts = []
        for i, m in enumerate(matches, 1):
            header = f"[Call #{i}: {m['call_title']} | {m['call_date']} | Company: {m.get('customer_company', 'Unknown')}]"
            parts.append(f"{header}\n{m['text']}\n")
        return "\n---\n".join(parts)
    
    def _build_messages(self, question: str, context: str, history: list[ChatMessage]) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in history[-10:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": f"Based on the following call transcript excerpts, please answer this question:\n\nQuestion: {question}\n\nRelevant Call Transcripts:\n{context}"})
        return messages
    
    async def query(self, request: ChatRequest) -> ChatResponse:
        start = time.time()
        enhanced = self._enhance_query(request.message)
        matches = await self.store.search(enhanced, self.retrieval_top_k, request.filters)
        if not matches:
            return ChatResponse(answer="I couldn't find any relevant call transcripts.", sources=[], query_time_ms=int((time.time()-start)*1000))
        top = matches[:self.rerank_top_k]
        context = self._format_context(top)
        messages = self._build_messages(request.message, context, request.conversation_history)
        resp = self.openai.chat.completions.create(model=self.chat_model, messages=messages, max_tokens=2000, temperature=0.7)
        sources = []
        seen = set()
        for m in top:
            if m["call_id"] not in seen:
                seen.add(m["call_id"])
                sources.append(SourceReference(call_id=m["call_id"], call_title=m["call_title"], call_date=m["call_date"], customer_company=m.get("customer_company"), relevance_score=m["score"], excerpt=m["text"][:200]+"..."))
        return ChatResponse(answer=resp.choices[0].message.content, sources=sources, query_time_ms=int((time.time()-start)*1000))
    
    async def query_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        enhanced = self._enhance_query(request.message)
        matches = await self.store.search(enhanced, self.retrieval_top_k, request.filters)
        if not matches:
            yield "I couldn't find any relevant call transcripts."
            return
        context = self._format_context(matches[:self.rerank_top_k])
        messages = self._build_messages(request.message, context, request.conversation_history)
        stream = self.openai.chat.completions.create(model=self.chat_model, messages=messages, max_tokens=2000, temperature=0.7, stream=True)
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

_pipeline: Optional[RAGPipeline] = None
def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None: _pipeline = RAGPipeline()
    return _pipeline
EOF

# main.py
cat > backend/app/main.py << 'EOF'
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import uuid, json, structlog
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, SyncRequest, SyncJob, SyncStatus, IndexStats
from app.gong_client import get_gong_client
from app.chunker import TranscriptChunker
from app.pinecone_store import get_pinecone_store
from app.rag_pipeline import get_rag_pipeline

log = structlog.get_logger()
sync_jobs: dict[str, SyncJob] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting_application")
    get_pinecone_store()
    get_rag_pipeline()
    yield

app = FastAPI(title="Gong RAG Chatbot API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return await get_rag_pipeline().query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in get_rag_pipeline().query_stream(request):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")

async def run_sync_job(job_id: str, request: SyncRequest):
    job = sync_jobs[job_id]
    job.status = SyncStatus.IN_PROGRESS
    try:
        gong, store, chunker = get_gong_client(), get_pinecone_store(), TranscriptChunker()
        async for call in gong.fetch_calls_with_transcripts(request.from_date, request.to_date, request.call_ids):
            chunks = chunker.chunk_call(call)
            await store.upsert_chunks(chunks)
            job.calls_processed += 1
            job.chunks_created += len(chunks)
        job.status = SyncStatus.COMPLETED
        job.completed_at = datetime.utcnow()
    except Exception as e:
        job.status = SyncStatus.FAILED
        job.errors.append(str(e))
        job.completed_at = datetime.utcnow()

@app.post("/api/sync", response_model=SyncJob)
async def start_sync(background_tasks: BackgroundTasks, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None, call_ids: Optional[str] = None):
    job_id = str(uuid.uuid4())
    request = SyncRequest(from_date=from_date, to_date=to_date, call_ids=call_ids.split(",") if call_ids else None)
    job = SyncJob(job_id=job_id, status=SyncStatus.PENDING, started_at=datetime.utcnow())
    sync_jobs[job_id] = job
    background_tasks.add_task(run_sync_job, job_id, request)
    return job

@app.get("/api/sync/{job_id}", response_model=SyncJob)
async def get_sync_status(job_id: str):
    if job_id not in sync_jobs: raise HTTPException(status_code=404, detail="Job not found")
    return sync_jobs[job_id]

@app.get("/api/stats", response_model=IndexStats)
async def get_stats():
    return await get_pinecone_store().get_stats()

@app.get("/api/example-queries")
async def get_example_queries():
    return {"queries": ["What are the most common use cases mentioned by prospective customers?", "Which competitors are most frequently mentioned in calls?", "What are the top objections to using Port?", "Which user personas are most involved from the customer side?", "What common themes do customers bring up?", "What are the top 5 industries in recent calls?"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Dockerfile
cat > backend/Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt')"
COPY app/ ./app/
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# ============================================
# FRONTEND FILES
# ============================================

cat > frontend/package.json << 'EOF'
{
  "name": "gong-rag-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "lucide-react": "^0.263.1",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0"
  }
}
EOF

cat > frontend/tailwind.config.js << 'EOF'
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: { extend: {} },
  plugins: [],
}
EOF

cat > frontend/postcss.config.js << 'EOF'
export default {
  plugins: { tailwindcss: {}, autoprefixer: {} },
}
EOF

cat > frontend/tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true
  },
  "include": ["src"]
}
EOF

cat > frontend/vite.config.ts << 'EOF'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default defineConfig({ plugins: [react()] })
EOF

cat > frontend/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gong RAG Chatbot</title>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/src/main.tsx"></script>
</body>
</html>
EOF

cat > frontend/src/main.tsx << 'EOF'
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
EOF

cat > frontend/src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;
EOF

cat > frontend/src/App.tsx << 'EOF'
import ChatInterface from './components/ChatInterface'
export default function App() { return <ChatInterface /> }
EOF

# ChatInterface.tsx (the main component)
cat > frontend/src/components/ChatInterface.tsx << 'EOF'
import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, ChevronDown, ChevronUp, Sparkles, MessageSquare } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const exampleQueries = [
  "What are the most common use cases mentioned by prospective customers?",
  "Which competitors are most frequently mentioned in calls?",
  "What are the top objections to using Port?",
  "Which user personas are most involved from the customer side?",
  "What common themes do customers bring up?",
  "What are the top 5 industries in recent calls?"
];

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  queryTime?: number;
  error?: boolean;
}

interface Source {
  call_id: string;
  call_title: string;
  call_date: string;
  customer_company?: string;
  relevance_score: number;
  excerpt: string;
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    const userMsg: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      const history = messages.map(m => ({ role: m.role, content: m.content }));
      const res = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, conversation_history: history })
      });
      if (!res.ok) throw new Error('Failed');
      const data = await res.json();
      setMessages(prev => [...prev, { role: 'assistant', content: data.answer, sources: data.sources, queryTime: data.query_time_ms }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.', error: true }]);
    } finally { setLoading(false); }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <header className="bg-slate-800/50 backdrop-blur-sm border-b border-slate-700/50 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/25">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-white">Call Insights AI</h1>
            <p className="text-sm text-slate-400">Powered by Gong transcript analysis</p>
          </div>
        </div>
      </header>
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-6 rounded-2xl bg-gradient-to-br from-violet-500/20 to-indigo-600/20 flex items-center justify-center border border-violet-500/30">
                <Sparkles className="w-8 h-8 text-violet-400" />
              </div>
              <h2 className="text-2xl font-semibold text-white mb-2">Ask about your sales calls</h2>
              <p className="text-slate-400 mb-8 max-w-md mx-auto">Get insights from customer conversations.</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-w-2xl mx-auto">
                {exampleQueries.map((q, i) => (
                  <button key={i} onClick={() => sendMessage(q)} className="text-left px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-700/50 hover:border-violet-500/50 hover:bg-slate-700/50 transition-all text-sm text-slate-300 hover:text-white">
                    <MessageSquare className="w-4 h-4 inline mr-2 text-violet-400" />{q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
              {msg.role === 'assistant' && <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center flex-shrink-0"><Bot className="w-5 h-5 text-white" /></div>}
              <div className={`max-w-2xl ${msg.role === 'user' ? 'order-first' : ''}`}>
                <div className={`rounded-2xl px-4 py-3 ${msg.role === 'user' ? 'bg-gradient-to-r from-violet-600 to-indigo-600 text-white' : msg.error ? 'bg-red-900/30 border border-red-700/50 text-red-200' : 'bg-slate-800/80 border border-slate-700/50 text-slate-100'}`}>
                  <p className="whitespace-pre-wrap">{msg.content}</p>
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-3">
                    <button onClick={() => setExpandedSources(p => ({...p, [idx]: !p[idx]}))} className="flex items-center gap-2 text-sm text-slate-400 hover:text-violet-400">
                      {expandedSources[idx] ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''} • {msg.queryTime}ms
                    </button>
                    {expandedSources[idx] && (
                      <div className="mt-3 space-y-2">
                        {msg.sources.map((src, sIdx) => (
                          <div key={sIdx} className="rounded-xl bg-slate-800/50 border border-slate-700/30 p-3">
                            <div className="flex items-start justify-between gap-2">
                              <div><p className="font-medium text-white text-sm">{src.call_title}</p><p className="text-xs text-slate-400 mt-0.5">{src.call_date} {src.customer_company && `• ${src.customer_company}`}</p></div>
                              <span className="text-xs px-2 py-0.5 rounded-full bg-violet-500/20 text-violet-300 border border-violet-500/30">{(src.relevance_score * 100).toFixed(0)}%</span>
                            </div>
                            <p className="text-xs text-slate-400 mt-2 line-clamp-2">{src.excerpt}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
              {msg.role === 'user' && <div className="w-8 h-8 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0"><User className="w-5 h-5 text-slate-300" /></div>}
            </div>
          ))}
          {loading && (
            <div className="flex gap-4">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center"><Bot className="w-5 h-5 text-white" /></div>
              <div className="rounded-2xl bg-slate-800/80 border border-slate-700/50 px-4 py-3">
                <div className="flex items-center gap-2 text-slate-400"><Loader2 className="w-4 h-4 animate-spin" /><span>Analyzing transcripts...</span></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>
      <footer className="bg-slate-800/50 backdrop-blur-sm border-t border-slate-700/50 px-4 py-4">
        <div className="max-w-4xl mx-auto flex gap-3">
          <input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendMessage(input)} placeholder="Ask about your sales calls..." disabled={loading} className="flex-1 bg-slate-900/50 border border-slate-700/50 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-violet-500/50 disabled:opacity-50" />
          <button onClick={() => sendMessage(input)} disabled={!input.trim() || loading} className="px-5 py-3 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 disabled:from-slate-700 disabled:to-slate-700 text-white rounded-xl font-medium flex items-center gap-2 disabled:cursor-not-allowed">
            <Send className="w-5 h-5" />
          </button>
        </div>
      </footer>
    </div>
  );
}
EOF

# docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      - GONG_ACCESS_KEY=${GONG_ACCESS_KEY}
      - GONG_ACCESS_KEY_SECRET=${GONG_ACCESS_KEY_SECRET}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
      - PINECONE_INDEX_NAME=${PINECONE_INDEX_NAME:-gong-transcripts}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment: ["VITE_API_URL=http://localhost:8000"]
    depends_on: [backend]
EOF

# README
cat > README.md << 'EOF'
# Gong RAG Chatbot

AI-powered analysis of Gong call transcripts for sales and marketing teams.

## Quick Start

1. Copy `.env.example` to `.env` and add your API keys
2. Start services: `docker-compose up --build`
3. Sync transcripts: `curl -X POST http://localhost:8000/api/sync`
4. Open http://localhost:3000

## API Endpoints

- `POST /api/chat` - Send a question
- `POST /api/sync` - Sync transcripts from Gong
- `GET /api/sync/{job_id}` - Check sync status
- `GET /api/stats` - Get index statistics
EOF

echo ""
echo "✅ Project created successfully!"
echo ""
echo "Next steps:"
echo "  cd gong-rag-chatbot"
echo "  cp backend/.env.example backend/.env"
echo "  # Edit backend/.env with your API keys"
echo "  docker-compose up --build"
echo ""
echo "Then sync your Gong transcripts:"
echo "  curl -X POST http://localhost:8000/api/sync"
