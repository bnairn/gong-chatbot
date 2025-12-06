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
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _format_entry(self, entry: TranscriptEntry) -> str:
        speaker_type = "Customer" if entry.speaker.is_customer else "Rep"
        return f"[{speaker_type} - {entry.speaker.name}]: {entry.text}"
    
    def _generate_chunk_id(self, call_id: str, chunk_index: int, text: str) -> str:
        return hashlib.md5(f"{call_id}:{chunk_index}:{text[:100]}".encode()).hexdigest()
    
    def _create_chunk(self, call: GongCall, entries: list[TranscriptEntry], chunk_index: int, total_chunks: int) -> TranscriptChunk:
        text = "\n".join(self._format_entry(e) for e in entries)
        speakers = list(set(e.speaker.name for e in entries))
        metadata = ChunkMetadata(call_id=call.metadata.call_id, call_title=call.metadata.title, call_date=call.metadata.date.strftime("%Y-%m-%d"), chunk_index=chunk_index, total_chunks=total_chunks, speakers_in_chunk=speakers, customer_company=call.metadata.customer_company, industry=call.metadata.industry, deal_stage=call.metadata.deal_stage, start_time=entries[0].start_time, end_time=entries[-1].end_time, has_customer_speech=any(e.speaker.is_customer for e in entries), has_rep_speech=any(not e.speaker.is_customer for e in entries))
        return TranscriptChunk(id=self._generate_chunk_id(call.metadata.call_id, chunk_index, text), text=text, metadata=metadata)
    
    def chunk_call(self, call: GongCall) -> list[TranscriptChunk]:
        if not call.transcript: return []
        turns, current_turn, current_speaker = [], [], None
        for entry in call.transcript:
            if entry.speaker.id != current_speaker and current_turn:
                turns.append(current_turn)
                current_turn = []
            current_speaker = entry.speaker.id
            current_turn.append(entry)
        if current_turn: turns.append(current_turn)
        
        chunks_data, current_entries, current_tokens = [], [], 0
        for turn in turns:
            turn_text = "\n".join(self._format_entry(e) for e in turn)
            turn_tokens = self._count_tokens(turn_text)
            if current_tokens + turn_tokens > self.chunk_size and current_entries:
                chunks_data.append(current_entries)
                overlap_entries, overlap_tokens = [], 0
                for e in reversed(current_entries):
                    e_tok = self._count_tokens(self._format_entry(e))
                    if overlap_tokens + e_tok <= self.chunk_overlap:
                        overlap_entries.insert(0, e)
                        overlap_tokens += e_tok
                    else: break
                current_entries, current_tokens = overlap_entries, overlap_tokens
            current_entries.extend(turn)
            current_tokens += turn_tokens
        if current_entries: chunks_data.append(current_entries)
        chunks_data = [c for c in chunks_data if self._count_tokens("\n".join(self._format_entry(e) for e in c)) >= self.min_chunk_size]
        chunks = [self._create_chunk(call, entries, idx, len(chunks_data)) for idx, entries in enumerate(chunks_data)]
        log.info("chunked_call", call_id=call.metadata.call_id, chunks_created=len(chunks))
        return chunks

def chunk_transcript(call: GongCall) -> list[TranscriptChunk]:
    return TranscriptChunker().chunk_call(call)
