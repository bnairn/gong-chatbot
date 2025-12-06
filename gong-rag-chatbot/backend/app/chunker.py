import tiktoken
import hashlib
import structlog
from app.config import get_settings
from app.models import GongCall, TranscriptChunk, ChunkMetadata, TranscriptEntry

log = structlog.get_logger()

class TranscriptChunker:
    def __init__(self):
        settings = get_settings()
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.min_chunk_size = settings.min_chunk_size
        # Hard limit for embedding model (text-embedding-3-small = 8191 tokens)
        self.max_embedding_tokens = 8000  # Leave buffer for safety
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _format_entry(self, entry: TranscriptEntry) -> str:
        speaker_type = "Customer" if entry.speaker.is_customer else "Rep"
        return f"[{speaker_type} - {entry.speaker.name}]: {entry.text}"
    
    def _generate_chunk_id(self, call_id: str, chunk_index: int, text: str) -> str:
        return hashlib.md5(f"{call_id}:{chunk_index}:{text[:100]}".encode()).hexdigest()
    
    def _split_oversized_chunk(self, entries: list[TranscriptEntry], call_id: str) -> list[list[TranscriptEntry]]:
        """Split a chunk that's too large into smaller valid chunks"""
        result = []
        current = []
        current_tokens = 0
        
        for entry in entries:
            entry_text = self._format_entry(entry)
            entry_tokens = self._count_tokens(entry_text)
            
            # If single entry is too large, truncate it
            if entry_tokens > self.max_embedding_tokens:
                log.warning(
                    "truncating_single_entry",
                    call_id=call_id,
                    entry_tokens=entry_tokens,
                    max_tokens=self.max_embedding_tokens
                )
                # Truncate this entry's text
                tokens = self.tokenizer.encode(entry.text)[:self.max_embedding_tokens - 100]
                entry.text = self.tokenizer.decode(tokens)
                entry_tokens = self._count_tokens(self._format_entry(entry))
            
            if current_tokens + entry_tokens > self.max_embedding_tokens and current:
                result.append(current)
                current = []
                current_tokens = 0
            
            current.append(entry)
            current_tokens += entry_tokens
        
        if current:
            result.append(current)
        
        return result
    
    def _create_chunk(self, call: GongCall, entries: list[TranscriptEntry], 
                      chunk_index: int, total_chunks: int) -> TranscriptChunk | None:
        text = "\n".join(self._format_entry(e) for e in entries)
        token_count = self._count_tokens(text)
        
        # Validate chunk is under embedding token limit
        if token_count > self.max_embedding_tokens:
            log.error(
                "chunk_exceeds_embedding_limit",
                call_id=call.metadata.call_id,
                chunk_index=chunk_index,
                tokens=token_count,
                max_allowed=self.max_embedding_tokens,
                text_preview=text[:200]
            )
            return None
        
        speakers = list(set(e.speaker.name for e in entries))
        metadata = ChunkMetadata(
            call_id=call.metadata.call_id,
            call_title=call.metadata.title,
            call_date=call.metadata.date.strftime("%Y-%m-%d"),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            speakers_in_chunk=speakers,
            customer_company=call.metadata.customer_company,
            industry=call.metadata.industry,
            deal_stage=call.metadata.deal_stage,
            start_time=entries[0].start_time,
            end_time=entries[-1].end_time,
            has_customer_speech=any(e.speaker.is_customer for e in entries),
            has_rep_speech=any(not e.speaker.is_customer for e in entries)
        )
        
        return TranscriptChunk(
            id=self._generate_chunk_id(call.metadata.call_id, chunk_index, text),
            text=text,
            metadata=metadata
        )
    
    def chunk_call(self, call: GongCall) -> list[TranscriptChunk]:
        if not call.transcript:
            return []
        
        # Group entries by speaker turns
        turns, current_turn, current_speaker = [], [], None
        for entry in call.transcript:
            if entry.speaker.id != current_speaker and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_speaker = entry.speaker.id
            current_turn.append(entry)
        if current_turn:
            turns.append(current_turn)
        
        # Create chunks from turns
        chunks_data, current_entries, current_tokens = [], [], 0
        for turn in turns:
            turn_text = "\n".join(self._format_entry(e) for e in turn)
            turn_tokens = self._count_tokens(turn_text)
            
            # Check if this single turn exceeds our chunk size
            if turn_tokens > self.chunk_size:
                log.warning(
                    "turn_exceeds_chunk_size",
                    call_id=call.metadata.call_id,
                    turn_tokens=turn_tokens,
                    chunk_size=self.chunk_size
                )
                # Save current chunk if exists
                if current_entries:
                    chunks_data.append(current_entries)
                    current_entries, current_tokens = [], 0
                
                # Split this oversized turn if it exceeds embedding limit
                if turn_tokens > self.max_embedding_tokens:
                    split_chunks = self._split_oversized_chunk(turn, call.metadata.call_id)
                    chunks_data.extend(split_chunks)
                else:
                    chunks_data.append(turn)
                continue
            
            if current_tokens + turn_tokens > self.chunk_size and current_entries:
                chunks_data.append(current_entries)
                
                # Add overlap from previous chunk
                overlap_entries, overlap_tokens = [], 0
                for e in reversed(current_entries):
                    e_tok = self._count_tokens(self._format_entry(e))
                    if overlap_tokens + e_tok <= self.chunk_overlap:
                        overlap_entries.insert(0, e)
                        overlap_tokens += e_tok
                    else:
                        break
                current_entries, current_tokens = overlap_entries, overlap_tokens
            
            current_entries.extend(turn)
            current_tokens += turn_tokens
        
        if current_entries:
            chunks_data.append(current_entries)
        
        # Filter by minimum chunk size
        chunks_data = [
            c for c in chunks_data 
            if self._count_tokens("\n".join(self._format_entry(e) for e in c)) >= self.min_chunk_size
        ]
        
        # Create chunk objects with validation
        chunks = []
        chunks_skipped = 0
        for idx, entries in enumerate(chunks_data):
            chunk = self._create_chunk(call, entries, idx, len(chunks_data))
            if chunk is not None:
                chunks.append(chunk)
            else:
                chunks_skipped += 1
                # Try to split and salvage if possible
                log.info(
                    "attempting_to_split_oversized_chunk",
                    call_id=call.metadata.call_id,
                    chunk_index=idx
                )
                split_chunks = self._split_oversized_chunk(entries, call.metadata.call_id)
                for split_idx, split_entries in enumerate(split_chunks):
                    split_chunk = self._create_chunk(
                        call, 
                        split_entries, 
                        idx * 1000 + split_idx,  # Unique index for splits
                        len(chunks_data)
                    )
                    if split_chunk is not None:
                        chunks.append(split_chunk)
        
        log.info(
            "chunked_call",
            call_id=call.metadata.call_id,
            chunks_created=len(chunks),
            chunks_skipped=chunks_skipped,
            total_chunks_attempted=len(chunks_data)
        )
        
        return chunks


def chunk_transcript(call: GongCall) -> list[TranscriptChunk]:
    return TranscriptChunker().chunk_call(call)