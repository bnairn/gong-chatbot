# Gong Sales Intelligence Chatbot

A RAG-powered chatbot that analyzes Gong sales call transcripts to surface insights about themes, objections, competitor mentions, and feature interests.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gong API      │────▶│  Transcript      │────▶│   ChromaDB      │
│   (gong_client) │     │  Chunking        │     │   Vector Store  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│   Streamlit     │◀────│   RAG Chain      │◀─────────────┘
│   Chat UI       │────▶│   (LangChain)    │
└─────────────────┘     └──────────────────┘
```

## Features

- **Gong Integration**: Fetches call transcripts via Gong's API
- **Vector Search**: Semantic search over transcripts using ChromaDB + OpenAI embeddings
- **RAG Pipeline**: Retrieval-augmented generation for accurate, grounded answers
- **Sales-Optimized Prompts**: Specialized prompts for sales intelligence queries
- **Chat Memory**: Multi-turn conversations with context
- **Quick Queries**: Pre-built buttons for common sales analysis questions
- **Source Attribution**: See which calls informed each answer

## Quick Start

### 1. Clone and Setup

```bash
cd gong-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.template .env
# Edit .env with your API keys
```

Required keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GONG_ACCESS_KEY` & `GONG_ACCESS_KEY_SECRET`: From Gong Settings > API

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Ingest Data

1. In the sidebar, click "Ingest Transcripts"
2. Use "mock data" checkbox for testing without Gong credentials
3. For real data, uncheck mock and ensure Gong credentials are set

## Example Queries

The chatbot excels at answering questions like:

- "What are the most common themes across initial sales calls?"
- "What objections come up most frequently?"
- "Which features are prospects most interested in?"
- "Which competitors are mentioned and what do prospects say about them?"
- "What are the main pain points prospects describe?"
- "What pricing concerns do prospects raise?"
- "What buying signals appear in the calls?"

## Project Structure

```
gong-chatbot/
├── app.py              # Streamlit chat interface
├── gong_client.py      # Gong API integration
├── vector_store.py     # ChromaDB vector database
├── rag_chain.py        # LangChain RAG pipeline
├── requirements.txt    # Python dependencies
├── .env.template       # Environment variable template
└── chroma_db/          # Vector store persistence (created on first run)
```

## Gong API Setup

1. Go to Gong Settings > API
2. Create a new API key
3. Note the Access Key and Access Key Secret
4. Add them to your `.env` file

API Documentation: https://gong.app.gong.io/settings/api/documentation

## Customization

### Using Pinecone Instead of ChromaDB

1. Add to requirements.txt:
   ```
   pinecone-client>=3.0.0
   langchain-pinecone>=0.0.3
   ```

2. Modify `vector_store.py` to use Pinecone:
   ```python
   from langchain_pinecone import PineconeVectorStore
   
   # Replace Chroma initialization with:
   vectorstore = PineconeVectorStore(
       index_name="gong-transcripts",
       embedding=self.embeddings
   )
   ```

### Adjusting Chunk Size

In `vector_store.py`, modify the `RecursiveCharacterTextSplitter`:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Larger for more context
    chunk_overlap=300,    # More overlap for continuity
)
```

### Using Different LLM Models

In `rag_chain.py`, change the model:

```python
self.llm = ChatOpenAI(
    model="gpt-4o-mini",  # Faster, cheaper
    # model="gpt-4-turbo", # More capable
)
```

## Troubleshooting

### "No data loaded"
Run the ingestion from the sidebar. Use mock data for testing.

### "Invalid API credentials"
Verify your Gong API keys are correct in `.env`

### Slow responses
- Reduce `k` in retriever settings for fewer chunks
- Use `gpt-4o-mini` instead of `gpt-4o`
- Consider adding caching to the vector store

### Out of memory
- Reduce chunk size
- Process transcripts in smaller batches
- Use Pinecone for cloud-hosted vector storage

## Development

Run with auto-reload:
```bash
streamlit run app.py --server.runOnSave true
```

Test individual components:
```bash
python gong_client.py   # Test Gong API
python vector_store.py  # Test vector store
python rag_chain.py     # Test RAG chain
```

## License

MIT
