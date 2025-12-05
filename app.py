"""
Streamlit Chat Interface for Gong Sales Intelligence.

A conversational UI for querying sales call transcripts.
"""

import os
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Gong Sales Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .quick-query-btn {
        margin: 0.25rem;
    }
    .stats-container {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_system():
    """Initialize the RAG system (cached for performance)."""
    from vector_store import TranscriptVectorStore
    from rag_chain import SalesCallRAG
    
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    vector_store = TranscriptVectorStore(persist_directory=persist_dir)
    rag = SalesCallRAG(vector_store=vector_store)
    
    return rag, vector_store


def ingest_from_gong(use_mock: bool = False, days_back: int = 30):
    """Fetch and ingest transcripts from Gong."""
    from vector_store import TranscriptVectorStore
    
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    if use_mock:
        from gong_client import MockGongClient
        client = MockGongClient()
    else:
        from gong_client import GongClient
        client = GongClient()
    
    with st.spinner("Fetching transcripts from Gong..."):
        from_date = datetime.now() - timedelta(days=days_back)
        transcripts = client.fetch_all_transcripts(from_date=from_date)
    
    with st.spinner("Ingesting into vector database..."):
        store = TranscriptVectorStore(persist_directory=persist_dir)
        count = store.ingest_transcripts(transcripts)
    
    return len(transcripts), count


def render_sidebar():
    """Render the sidebar with configuration and stats."""
    with st.sidebar:
        st.title("🎯 Gong Intelligence")
        st.markdown("---")
        
        # Quick queries section
        st.subheader("Quick Queries")
        
        quick_queries = {
            "🎯 Common Themes": "What are the most common themes across initial sales calls?",
            "🚫 Top Objections": "What are the most common objections raised by prospects?",
            "⭐ Hot Features": "Which features are prospects most interested in?",
            "🏢 Competitors": "Which competitors come up most frequently and what do prospects say about them?",
            "💡 Pain Points": "What are the main pain points prospects mention?",
            "💰 Pricing Concerns": "What pricing questions or concerns do prospects raise?",
            "⏰ Timeline/Urgency": "What do prospects say about their decision timeline?",
            "✅ Buying Signals": "What positive buying signals appear in the calls?"
        }
        
        for label, query in quick_queries.items():
            if st.button(label, key=f"quick_{label}", use_container_width=True):
                st.session_state.pending_query = query
        
        st.markdown("---")
        
        # Data management section
        st.subheader("📊 Data Management")
        
        # Show stats if available
        try:
            rag, vector_store = init_rag_system()
            stats = vector_store.get_stats()
            
            st.metric("Total Chunks", stats["total_chunks"])
            
            if stats["total_chunks"] == 0:
                st.warning("No data loaded yet. Ingest transcripts below.")
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        st.markdown("---")
        
        # Ingestion controls
        st.subheader("🔄 Ingest Data")
        
        use_mock = st.checkbox("Use mock data (for testing)", value=True)
        days_back = st.slider("Days of history", 7, 90, 30)
        
        if st.button("Ingest Transcripts", type="primary", use_container_width=True):
            try:
                transcript_count, chunk_count = ingest_from_gong(
                    use_mock=use_mock,
                    days_back=days_back
                )
                st.success(f"Ingested {transcript_count} calls ({chunk_count} chunks)")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Error during ingestion: {e}")
        
        st.markdown("---")
        
        # Clear history button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            try:
                rag, _ = init_rag_system()
                rag.clear_history()
            except Exception:
                pass
            st.rerun()
        
        st.markdown("---")
        st.caption("Built with LangChain + ChromaDB + OpenAI")


def render_sources(sources: list[dict]):
    """Render source cards for referenced calls."""
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)} calls referenced)", expanded=False):
        for source in sources:
            title = source.get("title", "Unknown Call")
            date = source.get("date", "")
            call_type = source.get("type", "")
            url = source.get("url", "")
            
            if date:
                try:
                    date_obj = datetime.fromisoformat(date.replace("Z", "+00:00"))
                    date = date_obj.strftime("%B %d, %Y")
                except Exception:
                    pass
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{title}**")
                st.caption(f"{call_type} • {date}")
            with col2:
                if url:
                    st.link_button("Open", url, use_container_width=True)


def main():
    """Main application entry point."""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("💬 Sales Call Intelligence")
    st.markdown("Ask questions about your sales calls to uncover insights, patterns, and opportunities.")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                render_sources(message["sources"])
    
    # Handle pending query from sidebar buttons
    if st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query(query)
    
    # Chat input
    if prompt := st.chat_input("Ask about your sales calls..."):
        process_query(prompt)


def process_query(query: str):
    """Process a user query and generate response."""
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            rag, vector_store = init_rag_system()
            
            # Check if we have data
            stats = vector_store.get_stats()
            if stats["total_chunks"] == 0:
                response = "I don't have any call transcripts loaded yet. Please ingest some data using the sidebar controls first."
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                return
            
            with st.spinner("Analyzing transcripts..."):
                result = rag.query(query, include_sources=True)
            
            st.markdown(result["answer"])
            
            if result.get("sources"):
                render_sources(result["sources"])
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", [])
            })
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an error: {str(e)}"
            })


if __name__ == "__main__":
    main()
