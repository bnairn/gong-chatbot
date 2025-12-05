#!/usr/bin/env python3
"""
CLI tool for ingesting Gong transcripts into the vector store.

Usage:
    python ingest.py --days 30              # Last 30 days of calls
    python ingest.py --mock                 # Use mock data for testing
    python ingest.py --clear --days 90      # Clear and reingest 90 days
"""

import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest Gong call transcripts into vector database"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days of call history to fetch (default: 30)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data for testing (no Gong credentials needed)"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingesting"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="Directory for vector store persistence"
    )
    parser.add_argument(
        "--call-types",
        type=str,
        nargs="+",
        help="Filter by call types (e.g., Discovery Demo)"
    )
    
    args = parser.parse_args()
    
    print(f"=" * 60)
    print("Gong Transcript Ingestion")
    print(f"=" * 60)
    print(f"Mode: {'Mock data' if args.mock else 'Live Gong API'}")
    print(f"Days: {args.days}")
    print(f"Persist dir: {args.persist_dir}")
    print(f"Clear first: {args.clear}")
    if args.call_types:
        print(f"Call types: {args.call_types}")
    print(f"=" * 60)
    
    # Initialize vector store
    from vector_store import TranscriptVectorStore
    
    store = TranscriptVectorStore(persist_directory=args.persist_dir)
    
    if args.clear:
        print("\nClearing existing data...")
        store.clear()
    
    # Get initial stats
    initial_stats = store.get_stats()
    print(f"\nInitial state: {initial_stats['total_chunks']} chunks")
    
    # Initialize client
    if args.mock:
        from gong_client import MockGongClient
        client = MockGongClient()
    else:
        from gong_client import GongClient
        client = GongClient()
    
    # Fetch transcripts
    print(f"\nFetching transcripts...")
    from_date = datetime.now() - timedelta(days=args.days)
    
    transcripts = client.fetch_all_transcripts(
        from_date=from_date,
        call_types=args.call_types
    )
    
    print(f"Found {len(transcripts)} transcripts")
    
    if not transcripts:
        print("No transcripts to ingest.")
        return
    
    # Ingest
    print("\nIngesting into vector store...")
    chunk_count = store.ingest_transcripts(transcripts)
    
    # Final stats
    final_stats = store.get_stats()
    print(f"\n{'=' * 60}")
    print("Ingestion Complete")
    print(f"{'=' * 60}")
    print(f"Transcripts processed: {len(transcripts)}")
    print(f"Chunks created: {chunk_count}")
    print(f"Total chunks in store: {final_stats['total_chunks']}")
    print(f"{'=' * 60}")
    
    # Show sample
    if transcripts:
        print("\nSample transcript titles:")
        for t in transcripts[:5]:
            print(f"  - {t['title']} ({t.get('type', 'Unknown')})")


if __name__ == "__main__":
    main()
