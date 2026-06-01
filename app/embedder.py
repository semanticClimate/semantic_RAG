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


def _embed_api(text: str) -> list[float]:
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

    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Requesting Hugging Face API embeddings (attempt {attempt + 1}) for: '{text[:40]}...'")
            response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": text},
                timeout=30
            )

            # Handle Hugging Face model loading state (503)
            if response.status_code == 503:
                try:
                    estimated_time = response.json().get("estimated_time", 10)
                except Exception:
                    estimated_time = 10
                logger.warning(f"Hugging Face model is loading. Waiting {estimated_time:.1f}s before retry...")
                time.sleep(min(estimated_time, 15))
                continue

            if response.status_code != 200:
                logger.error(f"Hugging Face API error {response.status_code}: {response.text}")
                raise RuntimeError(f"Hugging Face API returned status code {response.status_code}")

            res = response.json()

            # Handle different nested list formats returned by HF feature-extraction pipeline:
            # e.g., [0.1, 0.2, ...] or [[0.1, 0.2, ...]] or [[[0.1, 0.2, ...]]]
            vector = res
            while isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                vector = vector[0]

            if not isinstance(vector, list) or len(vector) == 0 or not isinstance(vector[0], (int, float)):
                raise RuntimeError(f"Unexpected response structure from Hugging Face: {res}")

            logger.debug(f"Hugging Face embedded text successfully ({len(text)} chars) -> vector dim {len(vector)}")
            return vector

        except requests.exceptions.RequestException as re:
            logger.error(f"Network error calling Hugging Face API (attempt {attempt + 1}): {re}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Network error calling Hugging Face API: {re}") from re
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error parsing Hugging Face API response: {e}")
            raise RuntimeError(f"Hugging Face embedding parsing failed: {e}") from e

    raise RuntimeError("Failed to retrieve embeddings from Hugging Face API after retries")


def embed(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    if HAS_LOCAL_TRANSFORMERS:
        try:
            logger.info("Using local SentenceTransformer library for embedding")
            return _embed_local(text)
        except Exception as e:
            logger.warning(f"Local embedding failed: {e}. Falling back to Hugging Face API...")

    logger.info("Using Hugging Face Serverless API for embedding")
    return _embed_api(text)