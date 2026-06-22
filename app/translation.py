import httpx

from config import Config
from app.logger import get_logger

logger = get_logger(__name__)


def translate_to_english(query: str, language: str) -> str:
    """
    Translate a user query to English before routing or retrieval.
    Falls back only if every configured translator fails.
    """
    if not query:
        return query
    if language == "English":
        return query

    messages = [
        {
            "role": "system",
            "content": (
                "Translate the following text to English. "
                "Output ONLY the English translation - no explanations, "
                "no notes, no punctuation changes."
            ),
        },
        {"role": "user", "content": query},
    ]

    errors = []

    for provider in _translation_providers():
        try:
            translated = provider(messages)
            if translated and translated.strip():
                translated = translated.strip()
                logger.info(f"Query translated ({language} -> English): '{translated[:80]}'")
                return translated
        except Exception as e:
            errors.append(str(e))
            logger.warning(f"Translation provider failed: {e}")

    logger.warning(
        f"Translation failed for language='{language}'. "
        f"Falling back to original query. Errors: {errors}"
    )
    return query


def _translation_providers():
    provider = Config.LLM_PROVIDER
    if provider not in {"auto", "ollama", "bedrock", "grok"}:
        provider = "auto"

    if provider == "bedrock":
        return [_translate_with_bedrock]
    if provider == "ollama":
        return [_translate_with_ollama]
    if provider == "grok":
        return [_translate_with_grok]

    providers = []
    if Config.BEDROCK_MODEL_ID:
        providers.append(_translate_with_bedrock)
    providers.append(_translate_with_ollama)
    if Config.GROK_API_KEY:
        providers.append(_translate_with_grok)
    return providers


def _translate_with_bedrock(messages: list[dict]) -> str:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
    response = client.converse(
        modelId=Config.BEDROCK_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": messages[1]["content"]}],
            }
        ],
        system=[{"text": messages[0]["content"]}],
        inferenceConfig={"temperature": 0.0, "maxTokens": 256},
    )
    return "".join(
        block.get("text", "")
        for block in response["output"]["message"]["content"]
        if isinstance(block, dict)
    )


def _translate_with_ollama(messages: list[dict]) -> str:
    import ollama

    client_kwargs = {"host": Config.OLLAMA_BASE_URL}
    if Config.OLLAMA_API_KEY:
        client_kwargs["headers"] = {"Authorization": f"Bearer {Config.OLLAMA_API_KEY}"}
    client = ollama.Client(**client_kwargs)
    response = client.chat(
        model=Config.OLLAMA_MODEL,
        messages=messages,
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


def _translate_with_grok(messages: list[dict]) -> str:
    if not Config.GROK_API_KEY:
        raise RuntimeError("Grok API key is not configured.")

    url = f"{Config.GROK_BASE_URL.rstrip('/')}/chat/completions"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {Config.GROK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": Config.GROK_MODEL,
            "messages": messages,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=Config.GROK_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
