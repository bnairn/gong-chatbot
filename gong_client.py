"""
Gong API Client for fetching call transcripts.

Gong API Documentation: https://gong.app.gong.io/settings/api/documentation

You'll need:
1. Access Key and Access Key Secret from Gong Settings > API
2. Base64 encode them as: base64(access_key:access_key_secret)
"""

import os
import base64
import requests
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class GongClient:
    """Client for interacting with Gong API to fetch call transcripts."""
    
    BASE_URL = "https://api.gong.io/v2"
    
    def __init__(self, access_key: Optional[str] = None, access_key_secret: Optional[str] = None):
        """
        Initialize Gong client with API credentials.
        
        Args:
            access_key: Gong API access key (or set GONG_ACCESS_KEY env var)
            access_key_secret: Gong API secret (or set GONG_ACCESS_KEY_SECRET env var)
        """
        self.access_key = access_key or os.getenv("GONG_ACCESS_KEY")
        self.access_key_secret = access_key_secret or os.getenv("GONG_ACCESS_KEY_SECRET")
        
        if not self.access_key or not self.access_key_secret:
            raise ValueError(
                "Gong API credentials required. Set GONG_ACCESS_KEY and "
                "GONG_ACCESS_KEY_SECRET environment variables or pass them directly."
            )
        
        # Create Basic Auth header
        credentials = f"{self.access_key}:{self.access_key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        self.headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make authenticated request to Gong API."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        if method == "GET":
            response = requests.get(url, headers=self.headers, params=data)
        else:
            response = requests.post(url, headers=self.headers, json=data)
        
        response.raise_for_status()
        return response.json()
    
    def get_calls(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        cursor: Optional[str] = None,
        limit: int = 100
    ) -> dict:
        """
        Fetch list of calls within a date range.
        
        Args:
            from_date: Start date for call filter (default: 30 days ago)
            to_date: End date for call filter (default: now)
            cursor: Pagination cursor for fetching more results
            limit: Number of calls to fetch per page (max 100)
        
        Returns:
            dict with 'calls' list and optional 'cursor' for pagination
        """
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()
        
        data = {
            "filter": {
                "fromDateTime": from_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "toDateTime": to_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
        }
        
        if cursor:
            data["cursor"] = cursor
        
        return self._make_request("POST", "calls", data)
    
    def get_call_transcripts(self, call_ids: list[str]) -> dict:
        """
        Fetch transcripts for specific calls.
        
        Args:
            call_ids: List of Gong call IDs (max 100 per request)
        
        Returns:
            dict with 'callTranscripts' containing transcript data
        """
        data = {
            "filter": {
                "callIds": call_ids[:100]  # API limit
            }
        }
        
        return self._make_request("POST", "calls/transcript", data)
    
    def get_call_details(self, call_id: str) -> dict:
        """
        Get detailed information about a specific call.
        
        Args:
            call_id: Gong call ID
        
        Returns:
            dict with call details including participants, duration, etc.
        """
        data = {
            "filter": {
                "callIds": [call_id]
            }
        }
        
        return self._make_request("POST", "calls/extensive", data)
    
    def fetch_all_transcripts(
        self,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        call_types: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Fetch all call transcripts within a date range.
        
        This handles pagination automatically and combines call metadata
        with transcript content.
        
        Args:
            from_date: Start date for calls
            to_date: End date for calls
            call_types: Filter by call type (e.g., ["Discovery", "Demo"])
        
        Returns:
            List of dicts with call metadata and transcript text
        """
        all_transcripts = []
        cursor = None
        
        print("Fetching calls from Gong...")
        
        while True:
            # Get batch of calls
            calls_response = self.get_calls(
                from_date=from_date,
                to_date=to_date,
                cursor=cursor
            )
            
            calls = calls_response.get("calls", [])
            if not calls:
                break
            
            # Filter by call type if specified
            if call_types:
                calls = [c for c in calls if c.get("type") in call_types]
            
            # Get call IDs for transcript fetch
            call_ids = [c["id"] for c in calls]
            
            if call_ids:
                # Fetch transcripts for these calls
                transcript_response = self.get_call_transcripts(call_ids)
                transcripts = transcript_response.get("callTranscripts", [])
                
                # Combine call metadata with transcripts
                call_map = {c["id"]: c for c in calls}
                
                for transcript in transcripts:
                    call_id = transcript["callId"]
                    call_meta = call_map.get(call_id, {})
                    
                    # Convert transcript segments to text
                    full_text = self._format_transcript(transcript)
                    
                    all_transcripts.append({
                        "call_id": call_id,
                        "title": call_meta.get("title", "Untitled Call"),
                        "date": call_meta.get("started"),
                        "duration": call_meta.get("duration"),
                        "type": call_meta.get("type"),
                        "participants": call_meta.get("parties", []),
                        "transcript": full_text,
                        "url": call_meta.get("url")
                    })
            
            # Check for more pages
            cursor = calls_response.get("records", {}).get("cursor")
            if not cursor:
                break
            
            print(f"  Fetched {len(all_transcripts)} transcripts so far...")
        
        print(f"Total transcripts fetched: {len(all_transcripts)}")
        return all_transcripts
    
    def _format_transcript(self, transcript: dict) -> str:
        """Format transcript segments into readable text with speaker labels."""
        segments = transcript.get("transcript", [])
        formatted_lines = []
        
        for segment in segments:
            speaker = segment.get("speakerName", "Unknown")
            text = " ".join([
                sentence.get("text", "") 
                for sentence in segment.get("sentences", [])
            ])
            
            if text.strip():
                formatted_lines.append(f"[{speaker}]: {text}")
        
        return "\n".join(formatted_lines)


# For testing without real Gong credentials
class MockGongClient:
    """Mock client for development/testing without Gong API access."""
    
    def fetch_all_transcripts(self, **kwargs) -> list[dict]:
        """Return sample transcripts for testing."""
        return [
            {
                "call_id": "mock_001",
                "title": "Discovery Call - Acme Corp",
                "date": "2024-01-15T10:00:00Z",
                "duration": 1800,
                "type": "Discovery",
                "participants": [
                    {"name": "John Sales", "affiliation": "Internal"},
                    {"name": "Jane Buyer", "affiliation": "External"}
                ],
                "transcript": """[John Sales]: Hi Jane, thanks for taking the time to meet today. I'd love to learn more about what you're looking for.
[Jane Buyer]: Thanks John. We're currently using Competitor X but we're frustrated with the lack of integrations.
[John Sales]: I hear that a lot. What specific integrations are most important to you?
[Jane Buyer]: We really need Salesforce and Slack integration. Our team lives in those tools.
[John Sales]: Great news - we have native integrations with both. Can you tell me more about your current workflow?
[Jane Buyer]: Sure. Our main pain point is that data gets siloed and we spend too much time on manual data entry.
[John Sales]: That's exactly what our automation features solve. How many people would be using this?
[Jane Buyer]: About 50 users across sales and customer success.
[John Sales]: Perfect. What's your timeline for making a decision?
[Jane Buyer]: We're hoping to implement something by Q2. Budget isn't a huge concern, but we need to see ROI within 6 months.""",
                "url": "https://app.gong.io/call/mock_001"
            },
            {
                "call_id": "mock_002",
                "title": "Demo - TechStart Inc",
                "date": "2024-01-16T14:00:00Z",
                "duration": 2700,
                "type": "Demo",
                "participants": [
                    {"name": "Sarah SE", "affiliation": "Internal"},
                    {"name": "Mike Tech", "affiliation": "External"}
                ],
                "transcript": """[Sarah SE]: Welcome Mike! Today I'll show you how our platform can streamline your sales process.
[Mike Tech]: Great. We're comparing you against Competitor Y and Competitor Z.
[Sarah SE]: Happy to help you see the differences. What features matter most to you?
[Mike Tech]: Reporting and analytics are huge for us. We need real-time dashboards.
[Sarah SE]: Let me show you our analytics suite. As you can see, everything updates in real-time.
[Mike Tech]: That's impressive. How does pricing work?
[Sarah SE]: We have per-seat pricing with volume discounts. For your team size, you'd be in our Growth tier.
[Mike Tech]: What about security? We're SOC 2 compliant and need vendors who are too.
[Sarah SE]: Absolutely - we're SOC 2 Type II certified and can provide our audit reports.
[Mike Tech]: Perfect. The interface looks much cleaner than Competitor Y. How long is implementation?
[Sarah SE]: Typical implementation is 2-4 weeks. We provide dedicated onboarding support.
[Mike Tech]: That's faster than what Competitor Z quoted. Let me take this back to the team.""",
                "url": "https://app.gong.io/call/mock_002"
            },
            {
                "call_id": "mock_003",
                "title": "Follow-up - GlobalCo",
                "date": "2024-01-17T09:00:00Z",
                "duration": 1200,
                "type": "Follow-up",
                "participants": [
                    {"name": "John Sales", "affiliation": "Internal"},
                    {"name": "Lisa Procurement", "affiliation": "External"}
                ],
                "transcript": """[John Sales]: Hi Lisa, following up on our demo last week. Have you had a chance to review the proposal?
[Lisa Procurement]: Yes, the team liked what they saw. We have a few concerns though.
[John Sales]: I'd love to address those. What's on your mind?
[Lisa Procurement]: The pricing is about 20% higher than our current solution.
[John Sales]: I understand. Let me explain the ROI our customers typically see. On average, teams save 10 hours per week.
[Lisa Procurement]: That's helpful. We're also worried about the learning curve.
[John Sales]: That's a common concern. We offer extensive training and our UI is designed to be intuitive.
[Lisa Procurement]: What about data migration? We have 5 years of data in our current system.
[John Sales]: We handle migration as part of onboarding at no extra cost. Takes about a week.
[Lisa Procurement]: Good to know. The other stakeholders want another demo with more team members.
[John Sales]: Absolutely. Let's schedule that. I can also connect you with a reference customer in your industry.
[Lisa Procurement]: That would be really helpful. We're mainly stuck on budget approval at this point.""",
                "url": "https://app.gong.io/call/mock_003"
            }
        ]


if __name__ == "__main__":
    # Test with mock client
    client = MockGongClient()
    transcripts = client.fetch_all_transcripts()
    
    for t in transcripts:
        print(f"\n{'='*60}")
        print(f"Call: {t['title']}")
        print(f"Date: {t['date']}")
        print(f"Type: {t['type']}")
        print(f"Transcript preview: {t['transcript'][:200]}...")
