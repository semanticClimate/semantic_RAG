import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY         = os.getenv("SECRET_KEY", "dev-secret")
    DEBUG              = os.getenv("FLASK_DEBUG", "1") == "1"

    # Redis
    REDIS_URL          = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # ChromaDB
    CHROMA_PATH        = os.getenv("CHROMA_PATH", "./chroma_db")
    CHROMA_COLLECTION  = os.getenv("CHROMA_COLLECTION", "climate_academy")

    # Embedding
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip().lower()
    EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    BEDROCK_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
    BEDROCK_EMBEDDING_DIMENSIONS = int(os.getenv("BEDROCK_EMBEDDING_DIMENSIONS", 1024))
    BEDROCK_EMBEDDING_RETRIES = int(os.getenv("BEDROCK_EMBEDDING_RETRIES", 10))
    BEDROCK_EMBEDDING_BASE_DELAY = float(os.getenv("BEDROCK_EMBEDDING_BASE_DELAY", 1.0))
    BEDROCK_EMBEDDING_SLEEP = float(os.getenv("BEDROCK_EMBEDDING_SLEEP", 0.2))

    # Ollama
    LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # AWS Bedrock
    AWS_REGION         = os.getenv("AWS_REGION", "us-east-1")
    BEDROCK_CHAT_MODEL = os.getenv("BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", 1000))
    BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", 0.3))

    # Chunking
    CHUNK_SIZE         = int(os.getenv("CHUNK_SIZE", 150))
    CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 30))
    CHUNK_MODE         = os.getenv("CHUNK_MODE", "default").strip().lower()
    SOURCE_HTML_PATH   = os.getenv("SOURCE_HTML_PATH", "input/climate_academy.html")
    SOURCE_HTML_PATHS  = [
        p.strip()
        for p in os.getenv("SOURCE_HTML_PATHS", "").split(",")
        if p.strip()
    ]

    # Retrieval
    TOP_K              = int(os.getenv("TOP_K", 5))
    DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", 0.7))

    # Session
    SESSION_TTL        = int(os.getenv("SESSION_TTL_SECONDS", 86400))

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
