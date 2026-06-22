# AWS EC2 Deployment Guide — Climate Academy RAG
**Setup: EC2 Ubuntu 22.04 · Nginx · Gunicorn · Celery · Redis · AWS Bedrock**

---

## Prerequisites Checklist

Before starting, make sure:
- [ ] You can SSH into your EC2 instance
- [ ] Your EC2 Security Group has **port 80 open** (HTTP inbound from `0.0.0.0/0`)
- [ ] Your EC2 Security Group has **port 22 open** (SSH)
- [ ] You have AWS credentials with Bedrock access and `meta.llama3-1-8b-instruct-v1:0` enabled

> **How to open port 80 on AWS:**
> Go to EC2 → Instances → click your instance → Security tab → click the Security Group →
> Edit Inbound Rules → Add Rule → Type: HTTP, Source: Anywhere (0.0.0.0/0) → Save

---

## Step 1 — SSH Into Your EC2 Instance

```bash
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
```

---

## Step 2 — Clone the Project

```bash
cd ~
git clone https://github.com/semanticClimate/semantic_RAG.git
cd semantic_RAG
```

---

## Step 3 — Install uv and Python Dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
```

Quick check:

```bash
uv run python -c "import flask, celery, redis, chromadb, boto3; print('All OK')"
```

---

## Step 4 — Install and Start Redis

```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
redis-cli ping
# Expected: PONG
```

---

## Step 5 — Configure Environment

```bash
cp .env.example .env
nano .env
```

Set these values (copy-paste and fill in your key):

```env
FLASK_DEBUG=0
SECRET_KEY=change-this-to-a-long-random-string

REDIS_URL=redis://localhost:6379/0

# AWS Bedrock
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=meta.llama3-1-8b-instruct-v1:0
BEDROCK_MAX_TOKENS=1024
BEDROCK_TEMPERATURE=0.3

CHROMA_PATH=./chroma_db
CHROMA_COLLECTION=climate_academy
EMBEDDING_MODEL=all-MiniLM-L6-v2

CHUNK_SIZE=150
CHUNK_OVERLAP=30
CHUNK_MODE=encyclopedia
SOURCE_HTML_PATH=input/climate_filtered.html

TOP_K=5
DISTANCE_THRESHOLD=0.7
SESSION_TTL_SECONDS=86400
LOG_LEVEL=INFO
```

Save and secure it:

```bash
chmod 600 .env
```

---

## Step 6 — Run Ingestion

```bash
uv run python ingest.py
```

Verify chunks loaded:

```bash
uv run python -c "
import chromadb
from config import Config
c = chromadb.PersistentClient(path=Config.CHROMA_PATH)
print('Chunks:', c.get_collection(Config.CHROMA_COLLECTION).count())
"
```

---

## Step 7 — Install Nginx

```bash
sudo apt install nginx -y
sudo systemctl enable nginx
sudo systemctl start nginx
```

Check it's running:

```bash
sudo systemctl status nginx
```

Test in your browser: `http://<your-ec2-public-ip>` — you should see the default Nginx welcome page.

---

## Step 8 — Configure Nginx

Create a config file for your app:

```bash
sudo nano /etc/nginx/sites-available/climate-academy
```

Paste this (replace `YOUR_EC2_PUBLIC_IP` with your actual IP):

```nginx
server {
    listen 80;
    server_name YOUR_EC2_PUBLIC_IP;

    # Forward all traffic to Gunicorn
    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # Timeout settings — important for LLM responses
        proxy_connect_timeout 60s;
        proxy_send_timeout    120s;
        proxy_read_timeout    120s;
    }
}
```

Enable it and disable the default site:

```bash
sudo ln -s /etc/nginx/sites-available/climate-academy /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
```

Test the config then reload:

```bash
sudo nginx -t
# Expected: syntax is ok / test is successful

sudo systemctl reload nginx
```

---

## Step 9 — Set Up Systemd Services

### Gunicorn service

```bash
sudo cp deploy/gunicorn.service.example /etc/systemd/system/gunicorn-climate.service
sudo nano /etc/systemd/system/gunicorn-climate.service
```

Make sure these lines match your setup:

```ini
[Unit]
Description=Gunicorn — Climate Academy
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/semantic_RAG
ExecStart=/home/ubuntu/.local/bin/uv run gunicorn \
    --workers 2 \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    "app:create_app()"
Restart=always
EnvironmentFile=/home/ubuntu/semantic_RAG/.env

[Install]
WantedBy=multi-user.target
```

### Celery service

```bash
sudo cp deploy/celery.service.example /etc/systemd/system/celery-climate.service
sudo nano /etc/systemd/system/celery-climate.service
```

Make sure it looks like:

```ini
[Unit]
Description=Celery — Climate Academy
After=network.target redis-server.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/semantic_RAG
ExecStart=/home/ubuntu/.local/bin/uv run celery \
    -A app.celery_app worker \
    --loglevel=info \
    --concurrency=2
Restart=always
EnvironmentFile=/home/ubuntu/semantic_RAG/.env

[Install]
WantedBy=multi-user.target
```

### Enable and start everything

```bash
sudo systemctl daemon-reload
sudo systemctl enable gunicorn-climate celery-climate
sudo systemctl start gunicorn-climate
sudo systemctl start celery-climate
```

---

## Step 10 — Verify Everything is Running

```bash
# Check all services at once
for svc in redis-server nginx gunicorn-climate celery-climate; do
    echo "$svc: $(sudo systemctl is-active $svc)"
done
```

All four should say `active`.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Test through Nginx (public):

```bash
curl http://<your-ec2-public-ip>/health
```

Quick chat test:

```bash
# 1. Create a session
curl -X POST http://127.0.0.1:8000/session

# 2. Send a message (replace SESSION_ID with value from step 1)
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"SESSION_ID","message":"What is climate change?","language":"English"}'

# 3. Poll for result (replace TASK_ID with value from step 2)
curl http://127.0.0.1:8000/result/TASK_ID
```

---

## Common Operations

### View logs

```bash
sudo journalctl -u gunicorn-climate -f
sudo journalctl -u celery-climate -f
sudo journalctl -u nginx -f
```

### Restart after code changes

```bash
cd ~/semantic_RAG
git pull
uv sync
sudo systemctl restart gunicorn-climate celery-climate
```

### Re-ingest after updating source HTML

```bash
uv run python ingest.py
sudo systemctl restart celery-climate
```

---

## Troubleshooting

| Problem | What to check |
|---|---|
| Port 80 not reachable | AWS Security Group — is HTTP inbound open? |
| `502 Bad Gateway` | Gunicorn not running — check `journalctl -u gunicorn-climate` |
| `curl /health` works but Nginx doesn't | Run `sudo nginx -t` and check `/etc/nginx/sites-enabled/` |
| Celery not processing | Check `journalctl -u celery-climate` and Redis with `redis-cli ping` |
| Bedrock returns access denied | EC2 role/user lacks `bedrock:InvokeModel` permission or model access is not enabled |
| Ollama returns 404 | Wrong `OLLAMA_MODEL` name — verify `gpt-oss:20b` is available on your plan |
