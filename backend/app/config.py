from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    gong_access_key: str
    gong_access_key_secret: str
    gong_base_url: str = "https://api.gong.io/v2"
    pinecone_api_key: str
    pinecone_index_name: str = "gong-transcripts"
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o"
    chunk_size: int = 6000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_embedding_tokens: int = 8000
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    max_context_tokens: int = 8000
    app_name: str = "Gong RAG Chatbot"
    debug: bool = False
    slack_signing_secret: Optional[str] = None
    slack_bot_token: Optional[str] = None
    slack_app_token: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_settings() -> Settings:
    return Settings()
