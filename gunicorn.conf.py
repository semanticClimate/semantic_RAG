# gunicorn.conf.py
import multiprocessing
from config import Config

# Socket — Nginx talks to Gunicorn through this file
bind            = "0.0.0.0:8000"

# Workers — standard formula is (2 × CPU cores) + 1
# For a free tier with 512MB RAM, we MUST restrict this to 1 worker
# otherwise we will run out of memory (OOM).
workers         = 1
worker_class    = "gevent"       # async worker — handles concurrent connections efficiently
worker_connections = 100         # max simultaneous connections per worker

# Preload app to save memory
preload_app     = True

# Timeouts
timeout         = 120            # worker killed if silent for 120s (enough for long polls)
keepalive       = 5              # keep connection alive for 5s between requests
graceful_timeout = 30            # time given to finish in-flight requests on shutdown

# Logging
accesslog       = "-"
errorlog        = "-"
loglevel        = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)sµs'

# Process naming — makes it easy to identify in htop/ps
proc_name       = "climate-rag-api"

# Security — don't expose Gunicorn version in headers
forwarded_allow_ips = "*"
proxy_protocol  = False
