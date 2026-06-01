#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Celery worker in the background..."
# Force concurrency=1 to prevent Celery from spawning multiple processes and crashing the 512MB instance
celery -A app.tasks.celery_app worker --concurrency=1 --loglevel=info &

echo "Starting Gunicorn server..."
gunicorn -b 0.0.0.0:$PORT run:app
