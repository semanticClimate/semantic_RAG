# Deployment Guide — Climate Academy RAG Backend

Complete step-by-step guide for deploying the backend on a Linux production server with Nginx, SSL, and systemd services.

All steps run **on the production server via SSH** unless stated otherwise.

---

## Before You Start — Checklist

- [ ] SSH access to the production server
- [ ] Server running Ubuntu 20.04+ or compatible Debian Linux
- [ ] Domain name (e.g. `api.yourorg.com`) pointing to the server's public IP
- [ ] Port 443 open on the server firewall
- [ ] GPU server (`172.16.41.42`) reachable from production server via SSH
- [ ] `llama3.1:8b` already pulled on the GPU server (`ollama list` confirms it)
- [ ] sudo access on the production server

---

## Step 1 — SSH Into the Server and Clone the Repo

```bash
ssh gylab204@your-server-ip

# Clone into home directory
cd ~
git clone https://github.com/semanticClimate/semantic_RAG.git
cd semantic_RAG
```

---

## Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version   # confirm installed
```

---

## Step 3 — Install Dependencies

```bash
uv sync
```

Confirm all imports work:
```bash
uv run python -c "import flask, celery, chromadb, ollama; print('All imports OK')"
```

---

## Step 4 — Install Redis

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping   # must return PONG
```

---

## Step 5 — Configure Environment

Generate a secure SECRET_KEY:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```


```bash
cp .env.example .env
nano .env
```

Set these values for production:

```env
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=<generate-a-random-string-see-below>

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
```

Restrict .env file permissions:
```bash
chmod 600 .env
```

---

## Step 6 — Set Up SSH Key Auth for Ollama Tunnel

The Ollama tunnel must run without a password prompt so systemd can manage it automatically.

```bash
# Generate SSH key on the production server
ssh-keygen -t ed25519 -C "climate-rag-ollama-tunnel"
# Press Enter for all prompts — no passphrase

# Copy key to the GPU server
ssh-copy-id gylab204@172.16.41.42

# Test passwordless login works
ssh -o BatchMode=yes gylab204@172.16.41.42 echo "Key auth OK"
# Must print: Key auth OK — no password prompt
```

---

## Step 7 — Test Ollama Tunnel Manually

```bash
# Start tunnel in background
ssh -L 11434:localhost:11434 gylab204@172.16.41.42 -N &

# Verify it hits the GPU server (not local)
curl http://localhost:11434/api/tags
# Must show llama3.1:8b in the models list

# Kill the background tunnel — systemd will manage it from now on
kill %1
```

---

## Step 8 — Set Up Log Directory

```bash
sudo mkdir -p /var/log/semantic_RAG
sudo chown $USER:$USER /var/log/semantic_RAG
```

---

## Step 9 — Run Ingestion

Ensure `input/climate_academy.html` is present:
```bash
ls input/climate_academy.html   # confirm file exists
```

Run ingestion:
```bash
uv run python ingest.py
```

Confirm chunk count:
```bash
uv run python -c "
import chromadb; from config import Config
c = chromadb.PersistentClient(path=Config.CHROMA_PATH)
print('Chunks indexed:', c.get_collection(Config.CHROMA_COLLECTION).count())
"
# Expected: 100+ chunks
```

---

## Step 10 — Install Nginx

```bash
sudo apt install nginx -y
sudo systemctl enable nginx
sudo systemctl start nginx
curl http://localhost   # should return Nginx welcome page
```

---

## Step 11 — Install Certbot

```bash
sudo apt install certbot python3-certbot-nginx -y
```

---

## Step 12 — Configure Nginx

```bash
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/semantic_RAG
sudo nano /etc/nginx/sites-available/semantic_RAG
```

Make these edits in the file:
- Replace `api.yourorg.com` with your actual domain name (appears 3 times)
- Replace `/path/to/semantic_RAG/static/` in the `/app` location block with your actual frontend path

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/semantic_RAG /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
# Expected: syntax is ok / test is successful
sudo systemctl reload nginx
```

---

## Step 13 — Get SSL Certificate

```bash
sudo certbot --nginx -d api.yourorg.com
# Enter your email when prompted
# Agree to terms of service
# Certbot automatically edits the Nginx config to add SSL
```

Test auto-renewal:
```bash
sudo certbot renew --dry-run
# Expected: All simulated renewals succeeded
```

---

## Step 14 — Set Up Systemd Services

### Gunicorn service

```bash
sudo cp deploy/gunicorn.service.example /etc/systemd/system/gunicorn-climate.service
sudo nano /etc/systemd/system/gunicorn-climate.service
```

Replace in the file:
- `/path/to/semantic_RAG` → actual project path e.g. `/home/gylab204/semantic_RAG`
- `gylab204` in `User=` and `Group=` → your actual username

### Celery service

```bash
sudo cp deploy/celery.service.example /etc/systemd/system/celery-climate.service
sudo nano /etc/systemd/system/celery-climate.service
```

Same replacements as above.

### Ollama tunnel service

```bash
sudo nano /etc/systemd/system/ollama-tunnel.service
```

Paste this (replace `gylab204` and GPU server IP with actual values):

```ini
[Unit]
Description=SSH tunnel to Ollama GPU server
After=network.target
Wants=network.target

[Service]
User=gylab204
ExecStart=/usr/bin/ssh \
    -N \
    -o ServerAliveInterval=60 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L 11434:localhost:11434 \
    gylab204@172.16.41.42
Restart=always
RestartSec=10s
SyslogIdentifier=ollama-tunnel

[Install]
WantedBy=multi-user.target
```

### Enable and start all services

```bash
# Reload systemd after adding new service files
sudo systemctl daemon-reload

# Enable all to start on boot
sudo systemctl enable ollama-tunnel
sudo systemctl enable gunicorn-climate
sudo systemctl enable celery-climate

# Start in dependency order
sudo systemctl start ollama-tunnel
sleep 5   # give tunnel time to connect
sudo systemctl start gunicorn-climate
sudo systemctl start celery-climate
```

---

## Step 15 — Verify Full Deployment

### Check all services are running

```bash
for svc in redis-server ollama-tunnel gunicorn-climate celery-climate nginx; do
  status=$(sudo systemctl is-active $svc)
  echo "$svc: $status"
done
# All should print: active
```

### Check Ollama tunnel hits the GPU server

```bash
curl http://localhost:11434/api/tags
# Must show llama3.1:8b — not {"models":[]}
```

### Test the full API over HTTPS

```bash
# Health check
curl https://api.yourorg.com/health
# Expected: {"status": "ok"}

# Create session
curl -X POST https://api.yourorg.com/session
# Expected: {"session_id": "..."}  — copy this value

# Send message
curl -X POST https://api.yourorg.com/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "PASTE-SESSION-ID", "message": "What is the greenhouse effect?"}'
# Expected: {"task_id": "..."}  — copy this value

# Poll result (repeat until status is "done")
curl https://api.yourorg.com/result/PASTE-TASK-ID
# Expected: {"status": "done", "answer": "...", "sources": [...]}
```

### Open the frontend

Navigate to `https://api.yourorg.com/app` in a browser.
In the API input bar, set the URL to `https://api.yourorg.com`.

---

## Post-Deployment Operations

### View live logs

```bash
sudo journalctl -u gunicorn-climate -f     # Flask/Gunicorn logs
sudo journalctl -u celery-climate -f       # Celery worker logs
sudo journalctl -u ollama-tunnel -f        # SSH tunnel logs
sudo tail -f /var/log/climate-rag/access.log  # HTTP access log
```

### Deploy code updates

```bash
cd ~/semantic_RAG
git pull origin main
uv sync
sudo systemctl restart gunicorn-climate
sudo systemctl restart celery-climate
```

### Update the book and re-ingest

```bash
# Replace input/climate_academy.html with the new version, then:
uv run python ingest.py
sudo systemctl restart celery-climate
```

### Check SSL certificate status

```bash
sudo certbot certificates
# Shows expiry date — auto-renews 30 days before expiry
```

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `/result` always returns `pending` | Celery worker not running | `sudo systemctl restart celery-climate` |
| `model not found` error | Ollama tunnel not active or model not on GPU server | `sudo systemctl restart ollama-tunnel` then `curl localhost:11434/api/tags` |
| `502 Bad Gateway` | Gunicorn not running or Unix socket missing | `sudo systemctl restart gunicorn-climate` |
| `503 service_unavailable` on `/session` | Redis not running | `sudo systemctl restart redis-server` |
| `ssl_error` in browser | SSL certificate issue | `sudo certbot renew` |
| Changes not live after `git pull` | Services not restarted | `sudo systemctl restart gunicorn-climate celery-climate` |
| Only 6 chunks indexed | Wrong HTML file used for ingestion | Replace `input/climate_academy.html` with full book, re-run `ingest.py` |
| Double log lines in Celery output | Logger propagation issue | Add `logger.propagate = False` in `app/logger.py` |

---

## Security Checklist

Complete these before considering the deployment production-ready:

- [ ] `FLASK_DEBUG=0` confirmed in production `.env`
- [ ] `SECRET_KEY` is a long random string
- [ ] `.env` has permissions `600` (`chmod 600 .env`)
- [ ] Port 11434 (Ollama) is blocked externally — only accessible via tunnel
- [ ] Port 6379 (Redis) is blocked externally — localhost only
- [ ] Nginx rate limiting confirmed active
- [ ] SSL certificate valid and auto-renewing
- [ ] API key authentication added to all endpoints (next development phase)