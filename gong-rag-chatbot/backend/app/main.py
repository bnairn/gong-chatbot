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
