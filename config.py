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
    EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Ollama
    LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    # Grok / xAI
    GROK_API_KEY       = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    GROK_BASE_URL      = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
    GROK_MODEL         = os.getenv("GROK_MODEL", "grok-4.3")
    GROK_TIMEOUT       = float(os.getenv("GROK_TIMEOUT_SECONDS", 120))

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
