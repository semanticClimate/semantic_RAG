#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting Celery worker in the background..."
celery -A app.tasks.celery_app worker --loglevel=info &

echo "Starting Gunicorn server..."
gunicorn -b 0.0.0.0:$PORT run:app
