"""
RAG Chain for sales call analysis.

Handles question answering over Gong call transcripts with
specialized prompts for sales intelligence queries.
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from vector_store import TranscriptVectorStore

load_dotenv()


# System prompt optimized for sales call analysis
SYSTEM_PROMPT = """You are an expert sales intelligence analyst with deep expertise in analyzing sales conversations. You have access to transcripts from sales calls and your job is to provide accurate, insightful analysis based on these transcripts.

When answering questions:
1. Base your answers ONLY on the provided call transcript excerpts
2. Quote specific examples from the transcripts when relevant
3. If the transcripts don't contain enough information to fully answer, acknowledge this
4. Provide structured, actionable insights when appropriate
5. Reference which calls your insights come from (by title/date when available)

For pattern/theme questions:
- Look across multiple transcripts to identify recurring patterns
- Quantify when possible (e.g., "mentioned in 3 of 5 calls")
- Organize findings by frequency or importance

For competitor questions:
- Note specific competitor names mentioned
- Capture what prospects say about competitors (positive and negative)
- Identify competitive advantages and disadvantages mentioned

For objection questions:
- Categorize objections (pricing, features, timing, etc.)
- Note how objections were handled
- Identify patterns in when objections arise

For feature/interest questions:
- List specific features mentioned with context
- Note the prospect's tone/enthusiasm level
- Connect features to underlying needs/pain points

Context from call transcripts:
{context}
"""

# Query refinement prompt for better retrieval
QUERY_REFINEMENT_PROMPT = """Given the user's question about sales calls, generate 2-3 search queries 
that would help find relevant transcript sections. Focus on:
- Key terms and synonyms
- Related concepts
- Specific phrases prospects might use

User question: {question}

Return only the search queries, one per line:"""


class SalesCallRAG:
    """RAG system for answering questions about sales call transcripts."""
    
    def __init__(
        self,
        vector_store: TranscriptVectorStore,
        model_name: str = "gpt-4o",
        temperature: float = 0.1
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_store: Initialized TranscriptVectorStore
            model_name: OpenAI model to use
            temperature: LLM temperature (lower = more focused)
        """
        self.vector_store = vector_store
        self.retriever = vector_store.get_retriever({"k": 10})
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # For query refinement
        self.query_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Build the RAG chain
        self.chain = self._build_chain()
        
        # Chat history for multi-turn conversations
        self.chat_history = []
    
    def _build_chain(self):
        """Build the RAG chain with retrieval and generation."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        def format_docs(docs: list[Document]) -> str:
            """Format retrieved documents for the prompt."""
            formatted = []
            for i, doc in enumerate(docs, 1):
                meta = doc.metadata
                header = f"--- Call {i}: {meta.get('title', 'Unknown')} ({meta.get('date', 'No date')}) ---"
                formatted.append(f"{header}\n{doc.page_content}")
            return "\n\n".join(formatted)
        
        # Chain: retrieve -> format -> generate
        chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.chat_history
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def _expand_query(self, question: str) -> list[str]:
        """Expand user query into multiple search queries for better retrieval."""
        prompt = QUERY_REFINEMENT_PROMPT.format(question=question)
        response = self.query_llm.invoke(prompt)
        queries = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        return [question] + queries  # Include original query
    
    def query(
        self,
        question: str,
        use_query_expansion: bool = True,
        include_sources: bool = True
    ) -> dict:
        """
        Ask a question about the sales calls.
        
        Args:
            question: User's question
            use_query_expansion: Whether to expand query for better retrieval
            include_sources: Whether to include source call info in response
        
        Returns:
            dict with 'answer' and optionally 'sources'
        """
        # Optionally expand query
        if use_query_expansion:
            queries = self._expand_query(question)
            all_docs = []
            seen_ids = set()
            
            for q in queries:
                docs = self.vector_store.search(q, k=5)
                for doc in docs:
                    doc_id = f"{doc.metadata['call_id']}_{doc.metadata['chunk_index']}"
                    if doc_id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(doc_id)
            
            # Sort by relevance (first query = most relevant)
            docs = all_docs[:12]
        else:
            docs = self.vector_store.search(question, k=10)
        
        # Format context
        context = self._format_context(docs)
        
        # Build prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT.format(context=context)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Keep history manageable
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        result = {"answer": answer}
        
        if include_sources:
            sources = self._extract_sources(docs)
            result["sources"] = sources
        
        return result
    
    def _format_context(self, docs: list[Document]) -> str:
        """Format documents into context string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = f"--- Call {i}: {meta.get('title', 'Unknown')} ({meta.get('date', 'No date')}) ---"
            formatted.append(f"{header}\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    def _extract_sources(self, docs: list[Document]) -> list[dict]:
        """Extract unique source information from documents."""
        seen = set()
        sources = []
        
        for doc in docs:
            call_id = doc.metadata.get("call_id")
            if call_id not in seen:
                seen.add(call_id)
                sources.append({
                    "call_id": call_id,
                    "title": doc.metadata.get("title"),
                    "date": doc.metadata.get("date"),
                    "type": doc.metadata.get("type"),
                    "url": doc.metadata.get("url")
                })
        
        return sources
    
    def clear_history(self):
        """Clear chat history for a fresh conversation."""
        self.chat_history = []
    
    def ask(self, question: str) -> str:
        """Simple query interface that returns just the answer."""
        result = self.query(question, include_sources=False)
        return result["answer"]


# Specialized query templates for common sales intelligence questions
QUERY_TEMPLATES = {
    "themes": "What are the most common themes, topics, or patterns that come up across initial sales calls?",
    "objections": "What are the most common objections or concerns raised by prospects during sales calls?",
    "features": "Which product features or capabilities are prospects most interested in or asking about?",
    "competitors": "Which competitors are mentioned in sales calls and what do prospects say about them?",
    "pain_points": "What are the main pain points or challenges that prospects mention?",
    "buying_signals": "What buying signals or positive indicators appear in the sales calls?",
    "pricing": "What questions or concerns do prospects raise about pricing?",
    "timeline": "What do prospects say about their timeline or urgency for making a decision?",
}


def create_rag_system(
    persist_directory: str = "./chroma_db",
    model_name: str = "gpt-4o"
) -> SalesCallRAG:
    """
    Create a RAG system from an existing vector store.
    
    Args:
        persist_directory: Where the vector store is persisted
        model_name: OpenAI model to use
    
    Returns:
        Initialized SalesCallRAG
    """
    vector_store = TranscriptVectorStore(persist_directory=persist_directory)
    return SalesCallRAG(vector_store=vector_store, model_name=model_name)


if __name__ == "__main__":
    # Test the RAG system
    from gong_client import MockGongClient
    from vector_store import create_vectorstore_from_transcripts
    
    # Create mock data
    client = MockGongClient()
    transcripts = client.fetch_all_transcripts()
    
    # Create vector store
    store = create_vectorstore_from_transcripts(transcripts, "./test_chroma_db")
    
    # Create RAG system
    rag = SalesCallRAG(vector_store=store)
    
    # Test queries
    test_questions = [
        "What competitors are mentioned in the calls?",
        "What objections do prospects raise?",
        "What features are prospects most interested in?"
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}")
        result = rag.query(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {len(result['sources'])} calls referenced")
