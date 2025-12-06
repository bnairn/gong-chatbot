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
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
    
    def _get_embedding(self, text: str) -> list[float]:
        return self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        ).data[0].embedding
    
    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of texts with proper token-based truncation"""
        results = []
        max_tokens = 8000  # Safe limit for text-embedding-3-small (actual limit: 8191)
        
        for text in texts:
            try:
                # Count tokens properly using tiktoken
                token_count = self._count_tokens(text)
                
                if token_count > max_tokens:
                    log.warning(
                        "truncating_text_for_embedding",
                        original_tokens=token_count,
                        max_tokens=max_tokens,
                        text_preview=text[:100]
                    )
                    # Truncate by TOKENS, not characters
                    tokens = self.tokenizer.encode(text)[:max_tokens]
                    text = self.tokenizer.decode(tokens)
                
                resp = self.openai.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                results.append(resp.data[0].embedding)
                
            except Exception as e:
                log.error(
                    "embedding_failed",
                    error=str(e),
                    text_length=len(text),
                    token_count=self._count_tokens(text) if hasattr(self, 'tokenizer') else 0
                )
                # DON'T return zero vectors - re-raise the error so sync job fails properly
                raise
        
        return results
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if not hasattr(self, 'tokenizer'):
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        return len(self.tokenizer.encode(text))
    
    def _chunk_to_metadata(self, chunk: TranscriptChunk) -> dict:
        return {
            "call_id": chunk.metadata.call_id,
            "call_title": chunk.metadata.call_title,
            "call_date": chunk.metadata.call_date,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            "speakers": ",".join(chunk.metadata.speakers_in_chunk),
            "customer_company": chunk.metadata.customer_company or "",
            "industry": chunk.metadata.industry or "",
            "deal_stage": chunk.metadata.deal_stage or "",
            "start_time": chunk.metadata.start_time,
            "end_time": chunk.metadata.end_time,
            "has_customer_speech": chunk.metadata.has_customer_speech,
            "has_rep_speech": chunk.metadata.has_rep_speech,
            "text": chunk.text
        }
    
    async def call_exists(self, call_id: str) -> bool:
        """Check if a call has already been indexed"""
        try:
            results = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector
                top_k=1,
                filter={"call_id": {"$eq": call_id}},
                include_metadata=False
            )
            return len(results.matches) > 0
        except Exception as e:
            log.warning("call_exists_check_failed", call_id=call_id, error=str(e))
            return False
    
    async def upsert_chunks(self, chunks: list[TranscriptChunk], batch_size: int = 100):
        if not chunks:
            return
        log.info("upserting_chunks", count=len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = self._get_embeddings_batch([c.text for c in batch])
            vectors = [
                {"id": c.id, "values": emb, "metadata": self._chunk_to_metadata(c)}
                for c, emb in zip(batch, embeddings)
            ]
            self.index.upsert(vectors=vectors)
    
    async def search(self, query: str, top_k: int = 10, filters: Optional[dict] = None) -> list[dict]:
        query_emb = self._get_embedding(query)
        pc_filter = None
        if filters:
            pc_filter = {
                k: {"$in": v} if isinstance(v, list) else {"$eq": v}
                for k, v in filters.items()
            }
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True,
            filter=pc_filter
        )
        return [
            {
                "id": m.id,
                "score": m.score,
                "text": m.metadata.get("text", ""),
                "call_id": m.metadata.get("call_id"),
                "call_title": m.metadata.get("call_title"),
                "call_date": m.metadata.get("call_date"),
                "customer_company": m.metadata.get("customer_company"),
                "industry": m.metadata.get("industry"),
                "chunk_index": m.metadata.get("chunk_index"),
                "has_customer_speech": m.metadata.get("has_customer_speech")
            }
            for m in results.matches
        ]
    
    async def delete_by_call_id(self, call_id: str):
        self.index.delete(filter={"call_id": {"$eq": call_id}})
    
    async def get_stats(self) -> IndexStats:
        stats = self.index.describe_index_stats()
        return IndexStats(
            total_vectors=stats.total_vector_count,
            total_calls=0,
            date_range={},
            top_companies=[],
            top_industries=[]
        )


_store: Optional[PineconeStore] = None

def get_pinecone_store() -> PineconeStore:
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store