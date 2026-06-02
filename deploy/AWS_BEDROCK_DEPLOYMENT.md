# AWS Bedrock Deployment Notes

This app can run on a normal CPU EC2 instance because inference is delegated to Amazon Bedrock.

## Recommended AWS shape

- **Compute:** EC2 `t3.small` or `t3.medium` for Flask, Celery, ChromaDB, and Redis.
- **LLM:** Amazon Bedrock chat model via the Converse API.
- **Embeddings:** Amazon Titan Text Embeddings V2 via Bedrock.
- **Vector DB:** local persistent ChromaDB on an attached EBS volume.
- **No GPU required:** do not run Ollama on AWS for this setup.

## Bedrock environment

Set these variables on the EC2 instance:

```bash
LLM_PROVIDER=bedrock
EMBEDDING_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_CHAT_MODEL=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
BEDROCK_EMBEDDING_DIMENSIONS=1024
BEDROCK_EMBEDDING_RETRIES=10
BEDROCK_EMBEDDING_BASE_DELAY=1.0
BEDROCK_EMBEDDING_SLEEP=0.2
BEDROCK_MAX_TOKENS=500
BEDROCK_CHAT_RETRIES=8
BEDROCK_CHAT_BASE_DELAY=2.0
```

Keep the existing Redis and Chroma variables:

```bash
REDIS_URL=redis://localhost:6379/0
CHROMA_PATH=/opt/climate-rag/chroma_db
CHROMA_COLLECTION=climate_academy
```

## Required AWS setup

1. Enable model access in the Bedrock console for the selected chat model and Titan embeddings.
2. Attach an IAM role to EC2 with permission for:
   - `bedrock:InvokeModel`
   - `bedrock:InvokeModelWithResponseStream`
3. Use the same `AWS_REGION` where those Bedrock models are enabled.

## Important ingestion step

If you switch from `sentence_transformers` to Bedrock embeddings, rebuild ChromaDB:

```bash
python ingest.py
```

Embeddings from different providers have different dimensions and cannot safely share the same Chroma collection.

## Quick smoke test

After deploying and starting Redis, Celery, and Gunicorn:

```bash
curl http://localhost:8000/health
```

Then create a session and send a chat message from the UI or API.
