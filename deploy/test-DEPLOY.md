# Lab Deployment Guide — Semantic RAG (Temporary / Internal)

> This is a temporary internal deployment for testing.
> Ollama on gpu001 is accessed directly over the lab network — no SSH tunnel needed.
> Internet-facing production deployment (with Nginx + SSL) comes later.

---

## Overview

```
172.16.41.41  (login-node)   — app runs here
  ├── Flask + Gunicorn        (port 8000)
  ├── Celery worker
  └── Redis (pre-installed, run from scratch)

172.16.41.42  (gpu001)       — LLM runs here
  └── ollama serve            (port 11434, direct network access)
```

---

## Phase 1 — Prepare gpu001 (Ollama)

**Location: your laptop — open a terminal and SSH into gpu001**

```bash
ssh gylab204@172.16.41.42
```

**Location: gpu001 terminal**

Check Ollama is installed and model is available:
```bash
which ollama
ollama --version
ollama list
# Must show llama3.1:8b
# If missing: ollama pull llama3.1:8b
```

Start Ollama in a tmux session so it survives SSH disconnection:
```bash
tmux new-session -d -s ollama "ollama serve"
```

Verify it is running and reachable:
```bash
sleep 3
curl http://localhost:11434/api/tags
# Must show llama3.1:8b in the models list
```

Verify it is reachable from the login-node side (run this from login-node later to confirm):
```bash
# Run this from login-node after SSH-ing into it:
curl http://172.16.41.42:11434/api/tags
# Must show llama3.1:8b
```

---

## Phase 2 — SSH Into login-node

**Location: your laptop — open a new terminal tab**

```bash
ssh gylab204@172.16.41.41
```

All remaining phases run on the login-node unless stated otherwise.

---

## Phase 3 — Clone the Repository

**Location: login-node**

Always work in scratch on HPC — never in home:
```bash
cd /lustre/scratch/gylab204/udita
```

Clone the repository:
```bash
git clone https://github.com/semanticClimate/semantic_RAG.git
cd semantic_RAG
```

Confirm the project structure is correct:
```bash
ls
# Must show: app/ config.py ingest.py run.py gunicorn.conf.py
#            html_sectioning.py input/ deploy/ index.html etc.
```

---

## Phase 4 — Install uv

**Location: login-node**

Check if already installed:
```bash
which uv
uv --version
```

If not installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version   # confirm
```

---

## Phase 5 — Create Virtual Environment and Install Dependencies

**Location: login-node, inside `/lustre/scratch/gylab204/udita/semantic_RAG`**

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG
uv sync
```

This reads `uv.lock` and installs everything into `.venv/` inside the project folder.
First run takes 3–5 minutes as packages download.

Verify all imports work:
```bash
uv run python -c "
import flask, celery, chromadb, ollama, sentence_transformers
print('All imports OK')
"
```

---

## Phase 6 — Configure Environment Variables

**Location: login-node, inside `semantic_RAG/`**

Generate a secure SECRET_KEY first:
```bash
uv run python -c "import secrets; print(secrets.token_hex(32))"
```

Copy the output — you will paste it below.

Create the `.env` file:
```bash
cp .env.example .env
nano .env
```

Set every value exactly as shown:
```env
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=paste-the-generated-key-here

REDIS_URL=redis://127.0.0.1:6379/0

CHROMA_PATH=/lustre/scratch/gylab204/udita/semantic_RAG/chroma_db
CHROMA_COLLECTION=climate_academy
EMBEDDING_MODEL=all-MiniLM-L6-v2

OLLAMA_BASE_URL=http://172.16.41.42:11434
OLLAMA_MODEL=llama3.1:8b

CHUNK_SIZE=150
CHUNK_OVERLAP=30
TOP_K=5
DISTANCE_THRESHOLD=0.7
SESSION_TTL_SECONDS=86400
LOG_LEVEL=INFO
```

Save and exit nano: `Ctrl+O` → Enter → `Ctrl+X`

Restrict permissions:
```bash
chmod 600 .env
```

---

## Phase 7 — Verify the Book File

**Location: login-node, inside `semantic_RAG/`**

```bash
ls input/
# Must show: climate_academy.html
```

If missing, transfer from your laptop:

**Location: your laptop (new terminal tab)**
```bash
scp /path/to/climate_academy.html \
  gylab204@172.16.41.41:/lustre/scratch/gylab204/udita/semantic_RAG/input/
```

---

## Phase 8 — Update gunicorn.conf.py

**Location: login-node, inside `semantic_RAG/`**

Gunicorn must bind to a TCP port (no Nginx, no Unix socket):
```bash
nano gunicorn.conf.py
```

Make these two changes:

**Change `bind`:**
```python
# From:
bind = "unix:/tmp/climate-rag.sock"
# To:
bind = "0.0.0.0:8000"
```

**Change log paths:**
```python
accesslog = "/lustre/scratch/gylab204/udita/semantic_RAG/logs/access.log"
errorlog  = "/lustre/scratch/gylab204/udita/semantic_RAG/logs/error.log"
```

Save and exit: `Ctrl+O` → Enter → `Ctrl+X`

Create the logs directory:
```bash
mkdir -p /lustre/scratch/gylab204/udita/semantic_RAG/logs
```

---

## Phase 9 — Verify routes.py Has the Index Route

**Location: login-node, inside `semantic_RAG/`**

Check that `app/routes.py` has the `/` route that serves `index.html`.
Open the file and confirm these lines exist near the top:

```bash
grep -n "send_from_directory\|def index" app/routes.py
```

If the route is missing, add it.

Open `app/routes.py`:
```bash
nano app/routes.py
```

Ensure the import line includes `send_from_directory`:
```python
from flask import Blueprint, request, jsonify, send_from_directory
import os
```

Add this route just above `@bp.route("/health")`:
```python
@bp.route("/")
def index():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return send_from_directory(project_root, "index.html")
```

Save and exit.

---

## Phase 10 — Start Redis

**Location: login-node**

Start Redis in a tmux session:
```bash
tmux new-session -d -s redis \
  "/lustre/scratch/gylab204/udita/redis-local/redis-stable/src/redis-server \
  --bind 127.0.0.1 --port 6379 \
  --save '' --appendonly no \
  --logfile /dev/null --daemonize no"
```

Verify Redis is running:
```bash
sleep 2
/lustre/scratch/gylab204/udita/redis-local/redis-stable/src/redis-cli ping
# Must return: PONG
```

---

## Phase 11 — Verify Ollama is Reachable from login-node

**Location: login-node**

```bash
curl http://172.16.41.42:11434/api/tags
# Must show llama3.1:8b in the models list
# If this fails, go back to Phase 1 and confirm Ollama is running on gpu001
```

---

## Phase 12 — Run Ingestion

**Location: login-node, inside `semantic_RAG/`**

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG
uv run python ingest.py
```

First run downloads the MiniLM model (~90MB) then embeds all chunks.
Watch the progress output — takes several minutes.

When complete, verify chunk count:
```bash
uv run python -c "
import chromadb
from config import Config
client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
col = client.get_collection(Config.CHROMA_COLLECTION)
print(f'Chunks indexed: {col.count()}')
"
# Expected: 100+ chunks
```

---

## Phase 13 — Start Celery Worker

**Location: login-node, inside `semantic_RAG/`**

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG

tmux new-session -d -s celery \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  uv run celery -A app.tasks worker \
  --concurrency=2 \
  --loglevel=info \
  --logfile=/lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log \
  -n climate-worker@%h"
```

Verify Celery started correctly:
```bash
sleep 4
tail -20 /lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log
# Must show: "celery@login-node ready"
```

---

## Phase 14 — Start Gunicorn (Flask)

**Location: login-node, inside `semantic_RAG/`**

```bash
tmux new-session -d -s gunicorn \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  uv run gunicorn 'app:create_app()' \
  --config gunicorn.conf.py \
  --env FLASK_ENV=production"
```

Verify Gunicorn started:
```bash
sleep 3
tail -20 /lustre/scratch/gylab204/udita/semantic_RAG/logs/gunicorn.log
# Must show: "Listening at: http://0.0.0.0:8000"
```

---

## Phase 15 — Verify All tmux Sessions Are Running

**Location: login-node**

```bash
tmux ls
```

Expected output:
```
celery:  1 windows (created ...)
gunicorn: 1 windows (created ...)
redis:   1 windows (created ...)

# On gpu001 (separate SSH session):
ollama:  1 windows (created ...)
```

To attach to any session and see live output:
```bash
tmux attach -t celery     # watch Celery logs live
tmux attach -t gunicorn   # watch Gunicorn logs live
# Detach without killing: Ctrl+B then D
```

---

## Phase 16 — Test the Full Deployment

**Location: login-node**

Run each command in sequence. Copy values between steps.

```bash
# 1. Health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# 2. Test frontend is served
curl -s http://localhost:8000/ | head -5
# Expected: first lines of HTML (<!DOCTYPE html> etc.)

# 3. Create session
curl -X POST http://localhost:8000/session
# Expected: {"session_id": "some-uuid-here"}
# Copy the session_id value

# 4. Send message (replace SESSION_ID with value from step 3)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "SESSION_ID", "message": "What is the greenhouse effect?"}'
# Expected: {"task_id": "some-task-id"}
# Copy the task_id value

# 5. Poll result (replace TASK_ID — run repeatedly until status is "done")
curl http://localhost:8000/result/TASK_ID
# First few calls: {"status": "pending"}
# After 20-60s:    {"status": "done", "answer": "...", "sources": [...]}
```

**From your laptop — confirm server is reachable on the lab network:**
```bash
curl http://172.16.41.41:8000/health
# Expected: {"status": "ok"}
```

**Open the frontend in a browser on your laptop:**
```
http://172.16.41.41:8000/
```

The page should load. In the API input bar at the bottom, confirm the URL shows `http://172.16.41.41:8000`.

---

## Phase 17 — Stopping All Services

**Location: login-node**

```bash
tmux kill-session -t gunicorn
tmux kill-session -t celery
tmux kill-session -t redis
```

**Location: gpu001**
```bash
tmux kill-session -t ollama
```

---

## Phase 18 — Restarting After SSH Timeout or Server Session Loss

If your SSH session drops and tmux sessions are lost, run this startup script.

First, create the script (only needed once):

**Location: login-node**
```bash
cat > /lustre/scratch/gylab204/udita/start_all.sh << 'EOF'
#!/bin/bash
set -e
BASE=/lustre/scratch/gylab204/udita
APP=$BASE/semantic_RAG
REDIS_BIN=$BASE/redis-local/redis-stable/src

echo "=== Checking Ollama on gpu001 ==="
curl -s http://172.16.41.42:11434/api/tags | grep -q "llama" \
  && echo "Ollama OK" \
  || echo "WARNING: Ollama not reachable — SSH into gpu001 and start it first"

echo ""
echo "=== Starting Redis ==="
tmux new-session -d -s redis \
  "$REDIS_BIN/redis-server \
  --bind 127.0.0.1 --port 6379 \
  --save '' --appendonly no \
  --logfile /dev/null --daemonize no" 2>/dev/null || echo "Redis session already exists"
sleep 2
$REDIS_BIN/redis-cli ping && echo "Redis OK" || echo "Redis FAILED"

echo ""
echo "=== Starting Celery ==="
tmux new-session -d -s celery \
  "cd $APP && uv run celery -A app.tasks worker \
  --concurrency=2 --loglevel=info \
  --logfile=$APP/logs/celery.log \
  -n climate-worker@%h" 2>/dev/null || echo "Celery session already exists"
sleep 4
tail -3 $APP/logs/celery.log

echo ""
echo "=== Starting Gunicorn ==="
tmux new-session -d -s gunicorn \
  "cd $APP && uv run gunicorn 'app:create_app()' \
  --config gunicorn.conf.py \
  --env FLASK_ENV=production" 2>/dev/null || echo "Gunicorn session already exists"
sleep 3
curl -s http://localhost:8000/health && echo "Gunicorn OK" || echo "Gunicorn FAILED"

echo ""
echo "=== All sessions ==="
tmux ls

echo ""
echo "=== Final health check ==="
curl http://localhost:8000/health
EOF

chmod +x /lustre/scratch/gylab204/udita/start_all.sh
```

Run it any time you need to restart:
```bash
bash /lustre/scratch/gylab204/udita/start_all.sh
```

Note: Ollama on gpu001 must be started manually first (Phase 1) before running this script.

---

## Viewing Logs

```bash
# Gunicorn (Flask request logs)
tail -f /lustre/scratch/gylab204/udita/semantic_RAG/logs/gunicorn.log

# Celery (RAG pipeline logs)
tail -f /lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log

# Attach to live tmux session
tmux attach -t celery
tmux attach -t gunicorn
# Detach: Ctrl+B then D
```

---

## Updating Code After a git push

**Location: login-node, inside `semantic_RAG/`**

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG
git pull origin main
uv sync   # only needed if pyproject.toml changed

# Restart Flask and Celery to pick up changes
tmux kill-session -t gunicorn
tmux kill-session -t celery

sleep 2

tmux new-session -d -s celery \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  uv run celery -A app.tasks worker \
  --concurrency=2 --loglevel=info \
  --logfile=/lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log \
  -n climate-worker@%h"

tmux new-session -d -s gunicorn \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  uv run gunicorn 'app:create_app()' \
  --config gunicorn.conf.py \
  --env FLASK_ENV=production"

sleep 3
curl http://localhost:8000/health
```

Redis does not need to restart unless `.env` Redis settings changed.
Celery does not need to restart unless the book was re-ingested.

---

## Final Checklist

```
Phase 1  — Ollama running on gpu001 (tmux session)          ✅ / ❌
Phase 2  — SSH into login-node                              ✅ / ❌
Phase 3  — Repo cloned to /lustre/scratch/gylab204/udita/   ✅ / ❌
Phase 4  — uv installed                                     ✅ / ❌
Phase 5  — uv sync complete, all imports OK                 ✅ / ❌
Phase 6  — .env configured, chmod 600                       ✅ / ❌
Phase 7  — climate_academy.html in input/ folder            ✅ / ❌
Phase 8  — gunicorn.conf.py using port 8000                 ✅ / ❌
Phase 9  — routes.py has / route for index.html             ✅ / ❌
Phase 10 — Redis running, ping returns PONG                 ✅ / ❌
Phase 11 — curl 172.16.41.42:11434/api/tags shows model     ✅ / ❌
Phase 12 — ingest.py complete, 100+ chunks indexed          ✅ / ❌
Phase 13 — Celery tmux session running, log shows ready     ✅ / ❌
Phase 14 — Gunicorn tmux session running, log shows 8000    ✅ / ❌
Phase 15 — tmux ls shows: redis, celery, gunicorn           ✅ / ❌
Phase 16 — /health OK, /session OK, /chat OK, /result OK    ✅ / ❌
Phase 16 — http://172.16.41.41:8000/ loads in browser       ✅ / ❌
```