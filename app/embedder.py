import os
from config import Config
from app.logger import get_logger
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = get_logger(__name__)

_model = None

def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading lightweight ONNX embedding model: {Config.EMBEDDING_MODEL}")
        try:
            # This automatically downloads and uses the all-MiniLM-L6-v2 ONNX model via AWS S3
            # It uses ~100MB RAM, avoids PyTorch, and bypasses Hugging Face DNS issues!
            _model = DefaultEmbeddingFunction()
            logger.info("ONNX Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX embedding model: {e}")
            raise RuntimeError(f"ONNX Embedding model could not be loaded: {e}") from e
    return _model

def _embed_local_onnx(text: str) -> list[float]:
    model = get_model()
    # DefaultEmbeddingFunction returns a list of embeddings
    vectors = model([text])
    return vectors[0]

def _embed_grok_api(text: str) -> list[float]:
    import requests
    if not Config.GROK_API_KEY:
        raise RuntimeError("GROK_API_KEY is not configured but Grok embedding provider was selected.")

    url = f"{Config.GROK_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {Config.GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": getattr(Config, "GROK_EMBEDDING_MODEL", "grok-embedding-small"),
        "input": text
    }

    try:
        logger.info(f"Requesting Grok embeddings for: '{text[:40]}...' using model {Config.GROK_EMBEDDING_MODEL}")
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            logger.error(f"Grok embedding API returned status code {response.status_code}: {response.text}")
            raise RuntimeError(f"Grok embedding API error {response.status_code}: {response.text}")

        res = response.json()
        vector = res["data"][0]["embedding"]
        logger.debug(f"Grok embedded text successfully -> vector dim {len(vector)}")
        return vector
    except Exception as e:
        logger.error(f"Grok embedding failed: {e}")
        raise RuntimeError(f"Grok embedding failed: {e}") from e


def embed(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    provider = Config.EMBEDDING_PROVIDER
    if provider == "grok":
        return _embed_grok_api(text)

    # For 'hf' or 'local', we use the ONNX model which requires no HuggingFace DNS
    logger.info("Using lightweight ONNX engine for embedding")
    return _embed_local_onnx(text)
