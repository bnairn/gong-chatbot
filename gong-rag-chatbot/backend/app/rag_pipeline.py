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
