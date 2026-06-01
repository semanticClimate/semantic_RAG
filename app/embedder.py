import os
import time
import requests
from config import Config
from app.logger import get_logger

logger = get_logger(__name__)

# Try importing sentence_transformers for local offline execution (e.g. local ingestion)
try:
    from sentence_transformers import SentenceTransformer
    HAS_LOCAL_TRANSFORMERS = True
except ImportError:
    HAS_LOCAL_TRANSFORMERS = False

_model = None

def get_model() -> "SentenceTransformer":
    global _model
    if not HAS_LOCAL_TRANSFORMERS:
        raise RuntimeError("SentenceTransformer is not installed. Use HF API.")
    if _model is None:
        logger.info(f"Loading embedding model locally: {Config.EMBEDDING_MODEL}")
        try:
            _model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model locally '{Config.EMBEDDING_MODEL}': {e}")
            raise RuntimeError(f"Embedding model could not be loaded: {e}") from e
    return _model


def _embed_local(text: str) -> list[float]:
    model = get_model()
    vector = model.encode(text, convert_to_numpy=True)
    return vector.tolist()


def _embed_hf_api(text: str) -> list[float]:
    # Use Hugging Face Serverless Inference API
    hf_token = os.getenv("HF_API_KEY")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    model_id = Config.EMBEDDING_MODEL
    # Resolve standard short name to Hugging Face repository name
    if "/" not in model_id:
        if model_id == "all-MiniLM-L6-v2":
            model_id = "sentence-transformers/all-MiniLM-L6-v2"

    # We try multiple official Hugging Face domains to bypass potential local DNS lookup issues on Render
    domains = [
        "https://api-inference.huggingface.co",
        "https://api.huggingface.co"
    ]

    last_error = None

    for domain in domains:
        api_url = f"{domain.rstrip('/')}/models/{model_id}"
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.info(f"Requesting Hugging Face API ({domain}) embeddings (attempt {attempt + 1}) for: '{text[:40]}...'")
                response = requests.post(
                    api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=20
                )

                # Handle Hugging Face model loading state (503)
                if response.status_code == 503:
                    try:
                        estimated_time = response.json().get("estimated_time", 10)
                    except Exception:
                        estimated_time = 10
                    logger.warning(f"Hugging Face model is loading on {domain}. Waiting {estimated_time:.1f}s before retry...")
                    time.sleep(min(estimated_time, 15))
                    continue

                if response.status_code != 200:
                    logger.warning(f"Hugging Face API ({domain}) returned error {response.status_code}: {response.text}")
                    last_error = RuntimeError(f"Hugging Face API returned status code {response.status_code}")
                    break  # Skip to the next domain immediately for non-503 errors

                res = response.json()

                # Handle different nested list formats returned by HF feature-extraction pipeline:
                vector = res
                while isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                    vector = vector[0]

                if not isinstance(vector, list) or len(vector) == 0 or not isinstance(vector[0], (int, float)):
                    logger.warning(f"Unexpected response structure from Hugging Face ({domain}): {res}")
                    last_error = RuntimeError(f"Unexpected response structure from Hugging Face")
                    break

                logger.debug(f"Hugging Face embedded text successfully via {domain} ({len(text)} chars) -> vector dim {len(vector)}")
                return vector

            except requests.exceptions.RequestException as re:
                logger.error(f"Network/DNS error calling Hugging Face API ({domain}) (attempt {attempt + 1}): {re}")
                last_error = re
                time.sleep(1)

    raise RuntimeError(f"Network error calling all Hugging Face API domains: {last_error}") from last_error


def _embed_grok_api(text: str) -> list[float]:
    if not Config.GROK_API_KEY:
        raise RuntimeError("GROK_API_KEY is not configured but Grok embedding provider was selected.")

    url = f"{Config.GROK_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {Config.GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": Config.GROK_EMBEDDING_MODEL,
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
    if provider == "local" or (provider == "hf" and HAS_LOCAL_TRANSFORMERS):
        try:
            logger.info("Using local SentenceTransformer library for embedding")
            return _embed_local(text)
        except Exception as e:
            if provider == "local":
                raise
            logger.warning(f"Local embedding failed: {e}. Falling back to Hugging Face API...")

    if provider == "grok":
        return _embed_grok_api(text)

    logger.info("Using Hugging Face Serverless API domains for embedding")
    return _embed_hf_api(text)
