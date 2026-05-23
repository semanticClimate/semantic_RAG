# Climate Academy RAG Chatbot — Backend

A production-grade Retrieval-Augmented Generation (RAG) API for the Climate Academy student book. Users ask questions and receive answers grounded strictly in the book's content, with inline section citations (§x.y.z).

---

## How It Works

1. The HTML book is parsed into sections, chunked, and embedded into a vector database (one-time ingestion)
2. A user sends a question via the frontend or API
3. Flask dispatches the question to a Celery background worker
4. The worker embeds the question, finds the most relevant book passages via ChromaDB, and sends them to Ollama (local LLM)
5. The LLM generates a grounded answer citing section numbers
6. The result is returned to the client via polling

---

## Architecture

```
Internet
    │  HTTPS :443
    ▼
  Nginx              reverse proxy, SSL, rate limiting
    │  Unix socket
    ▼
  Gunicorn           production WSGI server (4 async workers)
    │
  Flask              REST API — session management, task dispatch
    │
  Celery             async RAG pipeline execution
    │
  ┌─────────────────────────────────────────────┐
  │  Redis       message broker + session store  │
  │  ChromaDB    persistent vector database      │
  │  MiniLM      local embedding model           │
  │  Ollama      local LLM (Llama 3.1 8B / A100) │
  └─────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Web framework | Flask + Gunicorn | REST API server |
| Async tasks | Celery | RAG pipeline execution |
| Message broker | Redis | Flask ↔ Celery communication |
| Session store | Redis | Conversation history per user |
| Vector database | ChromaDB | Semantic chunk retrieval |
| Embedding model | all-MiniLM-L6-v2 | Local text → vector (384-dim) |
| LLM runtime | Ollama — Llama 3.1 8B | Answer generation on GPU |
| Reverse proxy | Nginx | SSL, rate limiting, routing |
| Book parser | BeautifulSoup4 | HTML → section records → chunks |
| Package manager | uv | Dependency management + venv |

---

## Project Structure

```
Climate-Chatbot/
├── app/
│   ├── __init__.py        Flask app factory + global error handlers
│   ├── routes.py          API endpoint definitions
│   ├── tasks.py           Celery task — RAG pipeline orchestration
│   ├── retriever.py       ChromaDB semantic search
│   ├── embedder.py        MiniLM embedding wrapper (singleton)
│   ├── llm.py             Ollama LLM call wrapper
│   ├── session.py         Redis session management
│   └── logger.py          Shared structured logging
├── deploy/
│   ├── nginx.conf.example          Reference Nginx config
│   ├── gunicorn.service.example    Reference systemd service
│   ├── celery.service.example      Reference systemd service
│   └── DEPLOY.md                   Full server deployment guide
├── input/
│   └── climate_academy.html         Source HTML book
├── html_sectioning.py     HTML parser → IndexedChunk objects
├── ingest.py              One-time ingestion pipeline script
├── config.py              All environment variable configuration
├── run.py                 Development server entry point
├── run_production.sh      Production startup script (used by systemd)
├── gunicorn.conf.py       Gunicorn production configuration
├── pyproject.toml         Project dependencies
├── uv.lock                Locked dependency versions
└── .env.example           Environment variable template
```

---

## API Endpoints

| Method | Endpoint | Description | Response |
|---|---|---|---|
| `GET` | `/health` | Health check | `{"status":"ok"}` |
| `POST` | `/session` | Create chat session | `{"session_id":"..."}` |
| `DELETE` | `/session/<id>` | Clear session + history | `{"message":"..."}` |
| `POST` | `/chat` | Send message | `{"task_id":"..."}` |
| `GET` | `/result/<task_id>` | Poll for answer | `{"status":"done","answer":"...","sources":[...]}` |

### Full Request Flow

```bash
# 1. Create session
curl -X POST http://localhost:5000/session
# → {"session_id": "abc-123"}

# 2. Send message
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "abc-123", "message": "What is the greenhouse effect?"}'
# → {"task_id": "xyz-456"}

# 3. Poll until done (status changes from "pending" to "done")
curl http://localhost:5000/result/xyz-456
# → {"status": "done", "answer": "The greenhouse effect...", "sources": [...]}
```

---

## Local Development Setup

### Prerequisites

| Tool | Install |
|---|---|
| Python 3.11+ | `sudo apt install python3` / `brew install python` |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Redis | `sudo apt install redis-server` / `brew install redis` |
| Ollama | `curl -fsSL https://ollama.com/install.sh \| sh` |

### Steps

```bash
# 1. Clone
git clone https://github.com/your-org/climate-rag-backend.git
cd climate-rag-backend

# 2. Install all dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# defaults work for local development — no changes needed

# 4. Start Redis
sudo systemctl start redis-server
redis-cli ping   # should return PONG

# 5. Pull the LLM model and start Ollama
ollama pull llama3.1:8b
ollama serve &

# 6. Run ingestion (once, or whenever the book changes)
uv run python ingest.py

# 7. Start services (four separate terminals)
uv run python run.py                                              # Terminal 1 — Flask
uv run celery -A app.tasks worker --concurrency=2 --loglevel=info # Terminal 2 — Celery

# 8. Test
curl http://localhost:5000/health
# → {"status": "ok"}
```

---

## Using a Remote Ollama Server (GPU)

If Ollama runs on a separate GPU server, use SSH port forwarding:

```bash
# Stop local Ollama to free the port
sudo systemctl stop ollama

# Open the tunnel — keep this terminal open
ssh -L 11434:localhost:11434 user@gpu-server-ip -N

# Verify the tunnel hits the GPU server (not local)
curl http://localhost:11434/api/tags
# Must show installed models — not {"models":[]}
```

No `.env` changes needed. The app calls `localhost:11434` which the tunnel forwards transparently.

> **Important:** Always start the SSH tunnel before starting Celery workers.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | Flask environment |
| `FLASK_DEBUG` | `1` | Debug mode (set to `0` in production) |
| `SECRET_KEY` | — | Flask secret key (change in production) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION` | `climate_academy` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name (must match `ollama list`) |
| `CHUNK_SIZE` | `150` | Words per chunk |
| `CHUNK_OVERLAP` | `30` | Overlap words between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `DISTANCE_THRESHOLD` | `0.7` | Max cosine distance for relevant chunks |
| `SESSION_TTL_SECONDS` | `86400` | Session expiry (24 hours) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG / INFO / WARNING) |

---

## Re-running Ingestion

Run whenever the source book changes:

```bash
uv run python ingest.py
sudo systemctl restart celery-climate   # restart workers to reload collection
```

Check indexed chunks:
```bash
uv run python -c "
import chromadb; from config import Config
c = chromadb.PersistentClient(path=Config.CHROMA_PATH)
print('Chunks:', c.get_collection(Config.CHROMA_COLLECTION).count())
"
```

---

## Logging

**Development:** all logs print to stdout with format:
```
2026-05-21 15:30:01 | INFO     | app.retriever | Retrieved 5 chunks — distance range: 0.31–0.58
```

**Production:** Gunicorn writes to `/var/log/climate-rag/`. View live:
```bash
sudo journalctl -u gunicorn-climate -f
sudo journalctl -u celery-climate -f
```

---

## Production Deployment

See [`deploy/DEPLOY.md`](deploy/DEPLOY.md) for the complete step-by-step server deployment guide.

---

## Known Constraints

| Constraint | Detail |
|---|---|
| LLM concurrency | Ollama handles one inference at a time — concurrent users queue via Celery |
| Vector DB scale | ChromaDB is single-node — suitable for ≤50 concurrent users |
| SSH tunnel | Remote Ollama requires an active SSH tunnel — automated via systemd in production |
| Chunk size | MiniLM max input is ~256 tokens ≈ 180 words — `CHUNK_SIZE=150` stays safely within limit |