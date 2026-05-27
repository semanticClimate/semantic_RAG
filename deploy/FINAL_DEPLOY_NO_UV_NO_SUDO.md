# Final Deployment Guide (No `uv`, No `sudo`)

This guide is for internal/lab deployment where:
- You do **not** have `sudo`
- You do **not** use `uv`
- You run everything in user space with `python -m venv`, `pip`, and `tmux`

---

## 1. Topology

- `172.16.41.41` (login-node): Flask + Gunicorn + Celery + Redis (user-space)
- `172.16.41.42` (gpu001): Ollama (`llama3.1:latest` or `llama3.1:8b`)

No Nginx, no SSL, no systemd.

---

## 2. Start Ollama on gpu001

On your laptop:

```bash
ssh gylab204@172.16.41.42
```

On gpu001:

```bash
ollama list
# ensure llama3.1 exists
# if needed: ollama pull llama3.1:latest

tmux new-session -d -s ollama "ollama serve"
sleep 2
curl http://localhost:11434/api/tags
```

---

## 3. Clone App on login-node

On your laptop:

```bash
ssh gylab204@172.16.41.41
```

On login-node:

```bash
cd /lustre/scratch/gylab204/udita
git clone https://github.com/semanticClimate/semantic_RAG.git
cd semantic_RAG
```

---

## 4. Python Env (No `uv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Quick check:

```bash
python -c "import flask, celery, chromadb, redis, sentence_transformers; print('imports ok')"
```

---

## 5. Configure `.env`

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
# paste output as SECRET_KEY
nano .env
```

Use values like:

```env
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=<your-generated-secret>

REDIS_URL=redis://127.0.0.1:6379/0

CHROMA_PATH=/lustre/scratch/gylab204/udita/semantic_RAG/chroma_db
CHROMA_COLLECTION=climate_academy
EMBEDDING_MODEL=all-MiniLM-L6-v2

OLLAMA_BASE_URL=http://172.16.41.42:11434
OLLAMA_MODEL=llama3.1:latest

CHUNK_SIZE=150
CHUNK_OVERLAP=30
CHUNK_MODE=default
SOURCE_HTML_PATH=input/climate_academy.html
SOURCE_HTML_PATHS=input/climate_academy.html,input/climate_filtered.html

TOP_K=5
DISTANCE_THRESHOLD=0.7
SESSION_TTL_SECONDS=86400
LOG_LEVEL=INFO
```

Then:

```bash
chmod 600 .env
mkdir -p logs
```

---

## 6. Gunicorn Config for No-Nginx

Edit `gunicorn.conf.py`:

```python
bind = "0.0.0.0:8000"
accesslog = "/lustre/scratch/gylab204/udita/semantic_RAG/logs/access.log"
errorlog = "/lustre/scratch/gylab204/udita/semantic_RAG/logs/error.log"
```

`0.0.0.0:8000` allows access from other lab machines.

---

## 7. Start Redis Without `sudo`

If your user-space Redis binary exists:

```bash
tmux new-session -d -s redis \
  "/lustre/scratch/gylab204/udita/redis-local/redis-stable/src/redis-server \
  --bind 127.0.0.1 --port 6379 --save '' --appendonly no --logfile /dev/null --daemonize no"

sleep 2
/lustre/scratch/gylab204/udita/redis-local/redis-stable/src/redis-cli ping
# PONG expected
```

If your cluster already provides a shared Redis service, point `REDIS_URL` to it and skip this step.

---

## 8. Ingest Data

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG
source .venv/bin/activate
python ingest.py
```

Verify:

```bash
python -c "import chromadb; from config import Config; c=chromadb.PersistentClient(path=Config.CHROMA_PATH); print(c.get_collection(Config.CHROMA_COLLECTION).count())"
```

---

## 9. Start Celery and Gunicorn in tmux

Celery:

```bash
tmux new-session -d -s celery \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  source .venv/bin/activate && \
  celery -A app.tasks worker --concurrency=2 --loglevel=info \
  --logfile=/lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log -n climate-worker@%h"
```

Gunicorn:

```bash
tmux new-session -d -s gunicorn \
  "cd /lustre/scratch/gylab204/udita/semantic_RAG && \
  source .venv/bin/activate && \
  gunicorn 'app:create_app()' --config gunicorn.conf.py --env FLASK_ENV=production"
```

Check sessions:

```bash
tmux ls
```

---

## 10. Validate End-to-End

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/session
```

Then test chat:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<SESSION_ID>","message":"What is the greenhouse effect?"}'

curl http://localhost:8000/result/<TASK_ID>
```

From laptop:

```bash
curl http://172.16.41.41:8000/health
```

---

## 11. Logs and Operations

```bash
tail -f /lustre/scratch/gylab204/udita/semantic_RAG/logs/error.log
tail -f /lustre/scratch/gylab204/udita/semantic_RAG/logs/celery.log
```

Stop services:

```bash
tmux kill-session -t gunicorn
tmux kill-session -t celery
tmux kill-session -t redis
```

Update code:

```bash
cd /lustre/scratch/gylab204/udita/semantic_RAG
git pull origin CABot
source .venv/bin/activate
pip install -r requirements.txt
# restart celery + gunicorn tmux sessions
```

---

## 12. Common Failures

- `tmux ls` says no server: the startup command crashed; run command in foreground once and fix error.
- `/result` always pending: Celery is not running or Redis is unreachable.
- `Connection refused` to `172.16.41.42:11434`: Ollama not running on gpu001.
- `gunicorn command not found`: virtual environment not activated.
- `module not found`: dependencies not installed in `.venv`.
