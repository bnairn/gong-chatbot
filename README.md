# Gong RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for analyzing Gong call transcripts using vector search and LLMs.

## Features

- Ingest and chunk Gong call transcripts
- Vector search using Pinecone
- Chat interface for querying call data
- Slack integration via slash commands

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r backend/requirements.txt`
3. Copy `.env` and fill in your API keys
4. Run the backend: `cd backend && python -m app.main`
5. Run the frontend: `cd frontend && npm install && npm run dev`

## Slack Integration

To integrate with Slack:

1. Create a Slack app at https://api.slack.com/apps
2. Add a slash command:
   - Command: `/gong-chat`
   - Request URL: `https://your-domain.com/api/slack/command` (or `http://localhost:8000/api/slack/command` for local testing)
   - Description: "Query Gong call transcripts"
3. Under "OAuth & Permissions", add bot token scopes:
   - `commands` (for slash commands)
4. Install the app to your workspace
5. Copy the following to your `.env` file:
   - `SLACK_SIGNING_SECRET`: From "Basic Information" > "Signing Secret"
   - `SLACK_BOT_TOKEN`: From "OAuth & Permissions" > "Bot User OAuth Token"
6. Restart the backend

### Usage

In Slack, type `/gong-chat What are the most common use cases mentioned by prospective customers?`

The bot will respond with an answer based on the indexed call transcripts, including source references.

## API Endpoints

- `GET /health` - Health check
- `POST /api/chat` - Chat with the bot
- `POST /api/chat/stream` - Streaming chat
- `POST /api/sync` - Sync Gong data
- `GET /api/sync/{job_id}` - Check sync status
- `GET /api/stats` - Index statistics
- `POST /api/slack/command` - Slack slash command handler

## Docker

Use `docker-compose up` to run the full stack.