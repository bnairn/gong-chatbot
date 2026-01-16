from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import uuid
import json
import structlog

from slack_sdk import WebClient
from slack_sdk.signature import SignatureVerifier

from app.config import get_settings
from app.models import ChatRequest, ChatResponse, SyncRequest, SyncJob, SyncStatus, IndexStats, SlackSlashCommand
from app.gong_client import get_gong_client
from app.chunker import TranscriptChunker
from app.pinecone_store import get_pinecone_store
from app.rag_pipeline import get_rag_pipeline

log = structlog.get_logger()

# In-memory storage for sync jobs
sync_jobs: dict[str, SyncJob] = {}

# Slack integration
slack_client: Optional[WebClient] = None
slack_verifier: Optional[SignatureVerifier] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global slack_client, slack_verifier
    log.info("starting_application")
    get_pinecone_store()
    get_rag_pipeline()
    
    settings = get_settings()
    if settings.slack_bot_token and settings.slack_signing_secret:
        slack_client = WebClient(token=settings.slack_bot_token)
        slack_verifier = SignatureVerifier(signing_secret=settings.slack_signing_secret)
        log.info("slack_integration_enabled")
    else:
        log.info("slack_integration_disabled")
    
    yield


app = FastAPI(title="Gong RAG Chatbot API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return await get_rag_pipeline().query(request)
    except Exception as e:
        log.error("chat_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        try:
            async for chunk in get_rag_pipeline().query_stream(request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            log.error("chat_stream_failed", error=str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


async def run_sync_job(job_id: str, request: SyncRequest):
    job = sync_jobs[job_id]
    job.status = SyncStatus.IN_PROGRESS
    skipped_count = 0
    
    try:
        gong = get_gong_client()
        store = get_pinecone_store()
        chunker = TranscriptChunker()
        
        log.info("sync_started", job_id=job_id)
        
        async for call in gong.fetch_calls_with_transcripts(
            request.from_date, 
            request.to_date, 
            request.call_ids
        ):
            # Skip if already indexed (deduplication)
            if await store.call_exists(call.metadata.call_id):
                log.info("skipping_already_indexed_call", call_id=call.metadata.call_id)
                skipped_count += 1
                continue
            
            # Chunk and index the call
            chunks = chunker.chunk_call(call)
            await store.upsert_chunks(chunks)
            
            job.calls_processed += 1
            job.chunks_created += len(chunks)
        
        job.status = SyncStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        
        log.info(
            "sync_completed",
            job_id=job_id,
            calls_processed=job.calls_processed,
            calls_skipped=skipped_count,
            chunks_created=job.chunks_created
        )
        
    except Exception as e:
        job.status = SyncStatus.FAILED
        job.errors.append(str(e))
        job.completed_at = datetime.utcnow()
        
        log.error(
            "sync_failed",
            job_id=job_id,
            error=str(e),
            calls_processed=job.calls_processed,
            calls_skipped=skipped_count
        )


@app.post("/api/sync", response_model=SyncJob)
async def start_sync(
    background_tasks: BackgroundTasks,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    call_ids: Optional[str] = None
):
    job_id = str(uuid.uuid4())
    
    request = SyncRequest(
        from_date=from_date,
        to_date=to_date,
        call_ids=call_ids.split(",") if call_ids else None
    )
    
    job = SyncJob(
        job_id=job_id,
        status=SyncStatus.PENDING,
        started_at=datetime.utcnow()
    )
    
    sync_jobs[job_id] = job
    background_tasks.add_task(run_sync_job, job_id, request)
    
    log.info("sync_job_created", job_id=job_id)
    
    return job


@app.get("/api/sync/{job_id}", response_model=SyncJob)
async def get_sync_status(job_id: str):
    if job_id not in sync_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return sync_jobs[job_id]


@app.get("/api/stats", response_model=IndexStats)
async def get_stats():
    return await get_pinecone_store().get_stats()


@app.get("/api/example-queries")
async def get_example_queries():
    return {
        "queries": [
            "What are the most common use cases mentioned by prospective customers?",
            "Which competitors are most frequently mentioned in calls?",
            "What are the top objections to using Port?",
            "Which user personas are most involved from the customer side?",
            "What common themes do customers bring up?",
            "What are the top 5 industries in recent calls?"
        ]
    }


@app.post("/api/slack/command")
async def slack_command(
    request: Request,
    token: str = Form(...),
    team_id: str = Form(...),
    team_domain: str = Form(...),
    channel_id: str = Form(...),
    channel_name: str = Form(...),
    user_id: str = Form(...),
    user_name: str = Form(...),
    command: str = Form(...),
    text: str = Form(...),
    response_url: str = Form(...),
    trigger_id: str = Form(...)
):
    if not slack_verifier:
        raise HTTPException(status_code=500, detail="Slack integration not configured")
    
    # Verify Slack signature
    body = await request.body()
    timestamp = request.headers.get("X-Slack-Request-Timestamp")
    signature = request.headers.get("X-Slack-Signature")
    
    if not slack_verifier.is_valid(body, timestamp, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    # Process the command
    if command != "/gong-chat":
        return {"text": "Unknown command"}
    
    if not text.strip():
        return {"text": "Please provide a question. Usage: /gong-chat What are the most common use cases?"}
    
    try:
        # Create chat request
        chat_request = ChatRequest(message=text.strip())
        
        # Get response from RAG pipeline
        response = await get_rag_pipeline().query(chat_request)
        
        # Format response for Slack
        sources_text = "\n\n".join([
            f"• {source.call_title} ({source.call_date}) - Relevance: {source.relevance_score:.2f}\n  {source.excerpt[:200]}..."
            for source in response.sources[:3]
        ])
        
        slack_response = {
            "response_type": "in_channel",
            "text": response.answer,
            "attachments": [
                {
                    "text": sources_text if sources_text else "No sources found"
                }
            ]
        }
        
        # Send response back to Slack
        import httpx
        async with httpx.AsyncClient() as client:
            await client.post(response_url, json=slack_response)
        
        # Return immediate response
        return {"text": "Processing your query..."}
        
    except Exception as e:
        log.error("slack_command_failed", error=str(e))
        return {"text": f"Sorry, an error occurred: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)