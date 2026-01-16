import httpx
import base64
from datetime import datetime
from typing import Optional, AsyncIterator
import structlog
from app.config import get_settings
from app.models import GongCall, CallMetadata, TranscriptEntry, Speaker

log = structlog.get_logger()

class GongClient:
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.gong_base_url
        creds = f"{settings.gong_access_key}:{settings.gong_access_key_secret}"
        encoded = base64.b64encode(creds.encode()).decode()
        self.headers = {"Authorization": f"Basic {encoded}", "Content-Type": "application/json"}
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        async with httpx.AsyncClient() as client:
            url = f"{self.base_url}{endpoint}"
            resp = await client.request(method, url, headers=self.headers, timeout=60.0, **kwargs)
            resp.raise_for_status()
            return resp.json()
    
    async def list_calls_extensive(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        cursor: Optional[str] = None
    ) -> tuple[list[dict], Optional[str]]:
        """List calls with extensive metadata using /v2/calls/extensive"""
        payload = {
            "filter": {},
            "contentSelector": {
                "exposedFields": {
                    "parties": True
                }
            }
        }
        
        if from_date:
            payload["filter"]["fromDateTime"] = from_date.isoformat() + "Z"
        if to_date:
            payload["filter"]["toDateTime"] = to_date.isoformat() + "Z"
        if cursor:
            payload["cursor"] = cursor
        
        data = await self._request("POST", "/calls/extensive", json=payload)
        calls = data.get("calls", [])
        next_cursor = data.get("records", {}).get("cursor")
        
        log.info("fetched_calls", count=len(calls), has_more=bool(next_cursor))
        return calls, next_cursor
    
    async def get_call_transcripts(self, call_ids: list[str]) -> dict:
        """Get transcripts for multiple calls. Returns dict mapping call_id -> transcript"""
        payload = {"filter": {"callIds": call_ids}}
        data = await self._request("POST", "/calls/transcript", json=payload)
        
        transcripts = {}
        for ct in data.get("callTranscripts", []):
            call_id = ct.get("callId")
            if call_id:
                transcripts[call_id] = ct.get("transcript", [])
        
        return transcripts
    
    def _parse_speaker(self, party: dict) -> Speaker:
        affiliation = party.get("affiliation", "")
        is_customer = affiliation.lower() in ["external"]
        
        return Speaker(
            id=str(party.get("speakerId", party.get("id", "unknown"))),
            name=party.get("name", "Unknown"),
            email=party.get("emailAddress"),
            is_customer=is_customer,
            title=party.get("title")
        )
    
    def _parse_call_metadata(self, call: dict) -> CallMetadata:
        metadata = call.get("metaData", {})
        parties = call.get("parties", [])
        
        participants = []
        customer_company = None
        
        for party in parties:
            speaker = self._parse_speaker(party)
            participants.append(speaker)
            if speaker.is_customer and party.get("company"):
                customer_company = party.get("company")
        
        started = metadata.get("started", "")
        if started:
            try:
                if started.endswith("Z"):
                    call_date = datetime.fromisoformat(started.replace("Z", "+00:00"))
                else:
                    call_date = datetime.fromisoformat(started)
            except:
                call_date = datetime.utcnow()
        else:
            call_date = datetime.utcnow()
        
        return CallMetadata(
            call_id=metadata.get("id", ""),
            title=metadata.get("title", "Untitled Call"),
            date=call_date,
            duration_seconds=metadata.get("duration", 0),
            participants=participants,
            customer_company=customer_company,
            industry=None,
            deal_stage=None,
            call_type=metadata.get("purpose"),
            url=metadata.get("url")
        )
    
    def _parse_transcript(self, raw: list[dict], speakers: list[Speaker]) -> list[TranscriptEntry]:
        # Build speaker map using speakerId
        speaker_map = {s.id: s for s in speakers if s.id and s.id != "None"}
        entries = []
        
        for segment in raw:
            speaker_id = str(segment.get("speakerId", "unknown"))
            speaker = speaker_map.get(speaker_id)
            
            if not speaker:
                speaker = Speaker(id=speaker_id, name=f"Speaker {speaker_id[-4:]}")
            
            sentences = segment.get("sentences", [])
            for sentence in sentences:
                text = sentence.get("text", "").strip()
                if text:
                    entries.append(TranscriptEntry(
                        speaker=speaker,
                        start_time=sentence.get("start", 0) / 1000,
                        end_time=sentence.get("end", 0) / 1000,
                        text=text
                    ))
        
        return entries
    
    async def fetch_calls_with_transcripts(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        call_ids: Optional[list[str]] = None
    ) -> AsyncIterator[GongCall]:
        all_calls = []
        
        if call_ids:
            payload = {
                "filter": {"callIds": call_ids},
                "contentSelector": {"exposedFields": {"parties": True}}
            }
            data = await self._request("POST", "/calls/extensive", json=payload)
            all_calls = data.get("calls", [])
        else:
            cursor = None
            while True:
                batch, cursor = await self.list_calls_extensive(from_date, to_date, cursor)
                all_calls.extend(batch)
                if not cursor:
                    break
        
        log.info("processing_calls", total=len(all_calls))
        
        batch_size = 20
        for i in range(0, len(all_calls), batch_size):
            batch = all_calls[i:i + batch_size]
            batch_ids = [call["metaData"]["id"] for call in batch]
            
            try:
                transcripts = await self.get_call_transcripts(batch_ids)
            except Exception as e:
                log.error("transcript_fetch_failed", batch_start=i, error=str(e))
                continue
            
            for call in batch:
                call_id = call["metaData"]["id"]
                transcript_raw = transcripts.get(call_id, [])
                
                if not transcript_raw:
                    log.debug("skipping_call_no_transcript", call_id=call_id)
                    continue
                
                try:
                    metadata = self._parse_call_metadata(call)
                    transcript = self._parse_transcript(transcript_raw, metadata.participants)
                    
                    if transcript:
                        yield GongCall(metadata=metadata, transcript=transcript)
                        log.info("processed_call", call_id=call_id, transcript_entries=len(transcript))
                except Exception as e:
                    log.error("call_parse_failed", call_id=call_id, error=str(e))
                    continue

_client: Optional[GongClient] = None

def get_gong_client() -> GongClient:
    global _client
    if _client is None:
        _client = GongClient()
    return _client