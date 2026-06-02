import json
import random
import time

from config import Config
from app.logger import get_logger

logger = get_logger(__name__)

_model = None
_bedrock_client = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
        try:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(Config.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{Config.EMBEDDING_MODEL}': {e}")
            raise RuntimeError(f"Embedding model could not be loaded: {e}") from e
    return _model


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        logger.info(f"Creating Bedrock Runtime client in {Config.AWS_REGION}")
        import boto3

        _bedrock_client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
    return _bedrock_client


def embed(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    if Config.EMBEDDING_PROVIDER == "bedrock":
        return _embed_with_bedrock(text)

    if Config.EMBEDDING_PROVIDER != "sentence_transformers":
        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER. Use 'sentence_transformers' or 'bedrock'."
        )

    try:
        model = get_model()
        vector = model.encode(text, convert_to_numpy=True)
        logger.debug(f"Embedded text ({len(text)} chars) -> vector dim {len(vector)}")
        return vector.tolist()
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise RuntimeError(f"Embedding failed: {e}") from e


def _embed_with_bedrock(text: str) -> list[float]:
    client = get_bedrock_client()
    body = json.dumps(
        {
            "inputText": text,
            "dimensions": Config.BEDROCK_EMBEDDING_DIMENSIONS,
            "normalize": True,
        }
    )

    for attempt in range(1, Config.BEDROCK_EMBEDDING_RETRIES + 1):
        try:
            response = client.invoke_model(
                modelId=Config.BEDROCK_EMBEDDING_MODEL,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(response["body"].read())
            vector = payload["embedding"]
            logger.debug(f"Embedded text with Bedrock ({len(text)} chars) -> vector dim {len(vector)}")
            if Config.BEDROCK_EMBEDDING_SLEEP > 0:
                time.sleep(Config.BEDROCK_EMBEDDING_SLEEP)
            return vector
        except Exception as e:
            error_name = e.__class__.__name__
            is_throttled = "Throttling" in error_name or "Too many requests" in str(e)
            if not is_throttled or attempt >= Config.BEDROCK_EMBEDDING_RETRIES:
                logger.error(f"Bedrock embedding failed: {e}")
                raise RuntimeError(f"Bedrock embedding failed: {e}") from e

            delay = min(Config.BEDROCK_EMBEDDING_BASE_DELAY * (2 ** (attempt - 1)), 60)
            delay += random.uniform(0, 0.5)
            logger.warning(
                f"Bedrock embedding throttled; retry {attempt}/"
                f"{Config.BEDROCK_EMBEDDING_RETRIES} in {delay:.1f}s"
            )
            time.sleep(delay)
