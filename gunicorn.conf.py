# gunicorn.conf.py
import multiprocessing
from config import Config

# Socket — Nginx talks to Gunicorn through this file
bind            = "0.0.0.0:8000"

# Workers — standard formula is (2 × CPU cores) + 1
# For a server with 4 cores this gives 9, but cap at 4 for our workload
# since the heavy work is in Celery workers, not Flask
workers         = 4
worker_class    = "gevent"       # async worker — handles concurrent connections efficiently
worker_connections = 100         # max simultaneous connections per worker

# Timeouts
timeout         = 120            # worker killed if silent for 120s (enough for long polls)
keepalive       = 5              # keep connection alive for 5s between requests
graceful_timeout = 30            # time given to finish in-flight requests on shutdown

# Logging
accesslog       = "/mnt/d/semantic_rag/semantic_RAG/logs/climate-rag/access.log"
errorlog        = "/mnt/d/semantic_rag/semantic_RAG/logs/climate-rag/error.log"
loglevel        = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# Process naming — makes it easy to identify in htop/ps
proc_name       = "climate-rag-api"

# Security — don't expose Gunicorn version in headers
forwarded_allow_ips = "*"
proxy_protocol  = False
