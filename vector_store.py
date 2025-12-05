"""
Vector Store for Gong call transcripts using ChromaDB.

Handles:
- Chunking transcripts for optimal retrieval
- Creating embeddings via OpenAI
- Storing and retrieving from vector database
"""

import os
import json
import hashlib
from typing import Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()

# Default persist directory for ChromaDB
DEFAULT_PERSIST_DIR = "./chroma_db"


class TranscriptVectorStore:
    """Vector store for Gong call transcripts with semantic search."""
    
    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = "gong_transcripts",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Where to store ChromaDB data
            collection_name: Name of the collection in ChromaDB
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Text splitter for chunking transcripts
        # Optimized for conversation structure
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n[", "\n", ". ", " ", ""],
            length_function=len
        )
        
        # Initialize or load existing vector store
        self._init_vectorstore()
    
    def _init_vectorstore(self):
        """Initialize or load the Chroma vector store."""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        print(f"Vector store initialized. Collection: {self.collection_name}")
    
    def _generate_doc_id(self, call_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a document chunk."""
        content = f"{call_id}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def ingest_transcripts(self, transcripts: list[dict], force_reingest: bool = False) -> int:
        """
        Ingest call transcripts into the vector store.
        
        Args:
            transcripts: List of transcript dicts from GongClient
            force_reingest: If True, reingest even if already exists
        
        Returns:
            Number of documents added
        """
        documents = []
        
        for transcript in transcripts:
            call_id = transcript["call_id"]
            
            # Skip if already ingested (unless forced)
            if not force_reingest and self._is_ingested(call_id):
                print(f"Skipping already ingested call: {call_id}")
                continue
            
            # Create chunks from transcript
            chunks = self.text_splitter.split_text(transcript["transcript"])
            
            # Create document for each chunk with metadata
            for i, chunk in enumerate(chunks):
                metadata = {
                    "call_id": call_id,
                    "title": transcript.get("title", ""),
                    "date": transcript.get("date", ""),
                    "type": transcript.get("type", ""),
                    "url": transcript.get("url", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "participants": json.dumps(transcript.get("participants", []))
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
        
        if documents:
            # Add to vector store
            self.vectorstore.add_documents(documents)
            print(f"Ingested {len(documents)} chunks from {len(transcripts)} transcripts")
        
        return len(documents)
    
    def _is_ingested(self, call_id: str) -> bool:
        """Check if a call has already been ingested."""
        try:
            results = self.vectorstore.similarity_search(
                "",
                k=1,
                filter={"call_id": call_id}
            )
            return len(results) > 0
        except Exception:
            return False
    
    def search(
        self,
        query: str,
        k: int = 10,
        call_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> list[Document]:
        """
        Search for relevant transcript chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            call_type: Filter by call type (e.g., "Discovery", "Demo")
            date_from: Filter calls from this date (ISO format)
            date_to: Filter calls until this date (ISO format)
        
        Returns:
            List of relevant Document objects
        """
        # Build filter if specified
        filter_dict = {}
        if call_type:
            filter_dict["type"] = call_type
        
        # ChromaDB doesn't support date range filters directly,
        # so we filter in post-processing if needed
        
        if filter_dict:
            results = self.vectorstore.similarity_search(
                query, k=k * 2, filter=filter_dict  # Fetch extra for date filtering
            )
        else:
            results = self.vectorstore.similarity_search(query, k=k * 2)
        
        # Apply date filtering if specified
        if date_from or date_to:
            filtered_results = []
            for doc in results:
                doc_date = doc.metadata.get("date", "")
                if doc_date:
                    if date_from and doc_date < date_from:
                        continue
                    if date_to and doc_date > date_to:
                        continue
                filtered_results.append(doc)
            results = filtered_results
        
        return results[:k]
    
    def search_with_scores(
        self,
        query: str,
        k: int = 10
    ) -> list[tuple[Document, float]]:
        """
        Search with relevance scores.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: dict = None):
        """
        Get a LangChain retriever for use in chains.
        
        Args:
            search_kwargs: Additional search parameters
        
        Returns:
            Retriever object
        """
        if search_kwargs is None:
            search_kwargs = {"k": 8}
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.vectorstore.delete_collection()
        self._init_vectorstore()
        print("Vector store cleared")


# Convenience function for quick setup
def create_vectorstore_from_transcripts(
    transcripts: list[dict],
    persist_directory: str = DEFAULT_PERSIST_DIR
) -> TranscriptVectorStore:
    """
    One-liner to create a vector store from transcripts.
    
    Args:
        transcripts: List of transcript dicts
        persist_directory: Where to persist the data
    
    Returns:
        Initialized and populated TranscriptVectorStore
    """
    store = TranscriptVectorStore(persist_directory=persist_directory)
    store.ingest_transcripts(transcripts)
    return store


if __name__ == "__main__":
    # Test with mock data
    from gong_client import MockGongClient
    
    client = MockGongClient()
    transcripts = client.fetch_all_transcripts()
    
    store = create_vectorstore_from_transcripts(transcripts, "./test_chroma_db")
    
    print("\nVector store stats:", store.get_stats())
    
    # Test search
    print("\n--- Testing search: 'competitor' ---")
    results = store.search("competitor", k=3)
    for doc in results:
        print(f"\n[{doc.metadata['title']}]")
        print(doc.page_content[:300])
