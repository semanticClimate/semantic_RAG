# Deployment Guide - Climate Academy RAG (No Nginx)

Complete step-by-step guide for deploying with:
- Flask app behind Gunicorn
- Celery workers
- Redis broker/backend
- Ollama (local or SSH tunnel)
- Frontend served directly by Flask route (`/`)

All steps run on the Linux server via SSH unless stated otherwise.

---

## Before You Start

- SSH access to the server
- Ubuntu 20.04+ (or compatible Debian)
- sudo access
- Ollama reachable (local or remote GPU host)

---

## Step 1 - Clone Project

```bash
ssh <user>@<server-ip>
cd ~
git clone https://github.com/semanticClimate/semantic_RAG.git
cd semantic_RAG
```

---

## Step 2 - Install uv and Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

Quick import check:

```bash
uv run python -c "import flask, celery, redis, chromadb, ollama; print('OK')"
```

---

## Step 3 - Install and Start Redis

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping
```

Expected output: `PONG`

---

## Step 4 - Configure Environment

```bash
cp .env.example .env
nano .env
```

Set at least:

```env
FLASK_DEBUG=0
SECRET_KEY=<random-secret>

REDIS_URL=redis://localhost:6379/0
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=climate_academy
EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=150
CHUNK_OVERLAP=30
TOP_K=5
DISTANCE_THRESHOLD=0.7
SESSION_TTL_SECONDS=86400
LOG_LEVEL=INFO

# New source/chunking controls
SOURCE_HTML_PATH=input/climate_filtered.html
CHUNK_MODE=encyclopedia
```

Secure `.env`:

```bash
chmod 600 .env
```

---

## Step 5 - Run Ingestion

```bash
uv run python ingest.py
```

Verify chunk count:

```bash
uv run python -c "import chromadb; from config import Config; c=chromadb.PersistentClient(path=Config.CHROMA_PATH); print(c.get_collection(Config.CHROMA_COLLECTION).count())"
```

---

## Step 6 - Ollama Setup

### Option A: Ollama on same server

```bash
ollama serve
ollama pull llama3.1:8b
curl http://localhost:11434/api/tags
```

### Option B: Ollama on remote GPU server (SSH tunnel)

```bash
ssh -N -L 11434:localhost:11434 <gpu-user>@<gpu-host>
```

Then verify:

```bash
curl http://localhost:11434/api/tags
```

---

## Step 7 - Systemd Services

Use included examples:

- `deploy/gunicorn.service.example`
- `deploy/celery.service.example`

Copy and edit placeholders:

```bash
sudo cp deploy/gunicorn.service.example /etc/systemd/system/gunicorn-climate.service
sudo cp deploy/celery.service.example /etc/systemd/system/celery-climate.service
sudo nano /etc/systemd/system/gunicorn-climate.service
sudo nano /etc/systemd/system/celery-climate.service
```

If using SSH tunnel for Ollama, add a third service (`ollama-tunnel.service`) similar to your existing deploy guide.

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable redis-server gunicorn-climate celery-climate
sudo systemctl start redis-server
sudo systemctl start gunicorn-climate
sudo systemctl start celery-climate
```

---

## Step 8 - Verify Everything

Check services:

```bash
for svc in redis-server gunicorn-climate celery-climate; do echo "$svc: $(sudo systemctl is-active $svc)"; done
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Frontend check (served by Flask route):

```bash
curl -I http://127.0.0.1:8000/
```

Chat flow check:

```bash
curl -X POST http://127.0.0.1:8000/session
curl -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d '{"session_id":"<id>","message":"What is climate change?","language":"English"}'
curl http://127.0.0.1:8000/result/<task_id>
```

---

## Runtime Flow

1. Browser opens `/` -> Flask serves `static/index.html`
2. `POST /chat` -> Flask enqueues Celery task in Redis
3. Celery retrieves passages from ChromaDB
4. Celery calls Ollama with grounded prompt
5. Result stored in Redis backend
6. Frontend polls `/result/<task_id>` until `done`

---

## Common Ops

Update code:

```bash
cd ~/semantic_RAG
git pull
uv sync
sudo systemctl restart gunicorn-climate celery-climate
```

Re-ingest after source update:

```bash
uv run python ingest.py
sudo systemctl restart celery-climate
```

Logs:

```bash
sudo journalctl -u gunicorn-climate -f
sudo journalctl -u celery-climate -f
sudo journalctl -u redis-server -f
```

