# chrome-ollama-live-agent

A starter project for a Chrome side panel extension that talks to a local Ollama model and can attach live web knowledge.

## Project structure

- `backend/`: Go API (`/chat`) that calls Ollama and optional web search tool
- `backend/`: Go API (`/chat`) with model providers, tools, and persisted conversations
- `extension/`: Chrome Extension Manifest V3 side panel UI + content script

## Prerequisites

- Chrome or Chromium
- Go 1.22+
- Ollama installed and running
- A local model pulled, for example:

```bash
ollama pull llama3.2
```

## Run backend

```bash
cd backend
go run .
```

Backend runs on `http://localhost:8080`.

Conversation persistence:
- Stored in backend file: `../data/chat_store.json` (relative to `backend/`)
- Override path with `CHAT_STORE_PATH=/absolute/or/relative/path.json`

### Provider keys (optional)

Set these only if you want cloud providers:

```bash
export OPENAI_API_KEY="your_openai_key"
export GEMINI_API_KEY="your_gemini_key"
export GEMINI_API_VERSION="v1beta"
```

Supported providers in the side panel:
- `ollama` (default, local)
- `openai` (requires `OPENAI_API_KEY`)
- `gemini` (requires `GEMINI_API_KEY`)

On backend startup, provider model lists are discovered and cached for:
- Ollama (local installed tags)
- OpenAI (`/v1/models`, filtered for chat-capable IDs)
- Gemini (`ListModels`, filtered for `generateContent`)

Health check:

```bash
curl http://localhost:8080/health
```

## Load extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `extension/` folder

## Use

1. Open any website
2. Click extension icon to open the side panel
3. Choose provider + model
4. Enter question in the bottom composer
5. Keep **Use live web knowledge** enabled for live lookup
6. Click **Ask**

## API contract

`POST /chat`

`GET /models`

`POST /models/refresh`

`GET /conversations`

`POST /conversations`

`GET /conversations/{id}`

`DELETE /conversations/{id}`

Request JSON:

```json
{
  "query": "What is this page about?",
  "provider": "ollama",
  "model": "llama3.2",
  "conversation_id": "conv_abc123",
  "use_live_web": true,
  "page_title": "...",
  "page_url": "...",
  "page_content": "..."
}
```

Response JSON:

```json
{
  "conversation_id": "conv_abc123",
  "answer": "...",
  "citations": [{"title": "...", "url": "..."}],
  "tool_calls": [{"name": "web_search", "input": "...", "status": "ok (3 results)"}]
}
```

`GET /models` response example:

```json
{
  "providers": {
    "ollama": {
      "ready": true,
      "models": ["llama3.2"],
      "last_refresh_unix": 1700000000,
      "error": ""
    },
    "openai": {
      "ready": false,
      "models": [],
      "last_refresh_unix": 1700000000,
      "error": "OPENAI_API_KEY is not set"
    },
    "gemini": {
      "ready": true,
      "models": ["gemini-2.0-flash"],
      "last_refresh_unix": 1700000000,
      "error": ""
    }
  }
}
```

## Notes

- The backend uses DuckDuckGo instant answer API for a keyless starter web search.
- For Gemini model-not-found errors, backend retries with discovered compatible Gemini models.
- Chats are persisted in backend and reloaded by the side panel conversation picker.
- For production, replace `webSearch` with a stronger provider (SerpAPI, Tavily, Brave Search) and add rate limits/auth.
