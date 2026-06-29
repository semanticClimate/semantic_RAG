import json
import re

import httpx

from config import Config
from app.logger import get_logger

logger = get_logger(__name__)


ROUTE_LABELS = {
    "metadata",
    "retrieval",
}

METADATA_INTENTS = {
    "book_title",
    "chapter_count",
    "chapter_list",
    "author",
    "edition_year",
    "climate_academy_started",
}

_METADATA_HINTS = [
    "chapter",
    "chapters",
    "chapter name",
    "chapter list",
    "author",
    "who wrote",
    "book title",
    "title of the book",
    "edition year",
    "published",
    "founded",
    "started",
    "climate academy",
]


def _normalize(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _route_prompt(query: str) -> str:
    return f"""You are a routing classifier for the Climate Academy assistant.

Classify the user's question into exactly one of these routes:
- metadata
- retrieval

Use metadata only for book-level facts such as:
- book title
- chapter count
- chapter list
- author
- edition year
- when Climate Academy started

Use retrieval for all other content questions.

Return ONLY valid JSON with this schema:
{{"route":"metadata|retrieval","intent":"one of book_title, chapter_count, chapter_list, author, edition_year, climate_academy_started, or empty string","confidence":0.0}}

Rules:
- If the question asks for chapters, chapter names, chapter list, or similar book metadata, choose metadata and intent chapter_list.
- If the question asks who wrote the book or the author, choose metadata and intent author.
- If the question asks when Climate Academy started or was founded, choose metadata and intent climate_academy_started.
- If the question asks what year the edition/book is, choose metadata and intent edition_year.
- If the question is asking for a chapter summary, specific chapter content, or about a particular topic/chapter (including climate anxiety), choose retrieval.
- If the question is about the content of the book, climate science, explanations, summaries, or passages, choose retrieval.

User question:
{query}
"""


def _parse_router_output(text: str) -> dict | None:
    text = _normalize(text)
    if not text:
        return None

    try:
        data = json.loads(text)
        route = str(data.get("route", "")).strip().lower()
        intent = str(data.get("intent", "")).strip().lower()
        confidence = float(data.get("confidence", 0.0))
        if route not in ROUTE_LABELS:
            return None
        if intent and intent not in METADATA_INTENTS:
            intent = ""
        return {"route": route, "intent": intent, "confidence": confidence}
    except Exception:
        return None


def _route_with_bedrock(query: str) -> dict | None:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
    response = client.converse(
        modelId=Config.BEDROCK_MODEL_ID,
        system=[{
            "text": "You output only valid JSON."
        }],
        messages=[{
            "role": "user",
            "content": [{"text": _route_prompt(query)}],
        }],
        inferenceConfig={"temperature": 0.0, "maxTokens": 128},
    )
    text = "".join(
        block.get("text", "")
        for block in response["output"]["message"]["content"]
        if isinstance(block, dict)
    ).strip()
    return _parse_router_output(text)


def _route_with_ollama(query: str) -> dict | None:
    import ollama

    client_kwargs = {"host": Config.OLLAMA_BASE_URL}
    if Config.OLLAMA_API_KEY:
        client_kwargs["headers"] = {"Authorization": f"Bearer {Config.OLLAMA_API_KEY}"}
    client = ollama.Client(**client_kwargs)
    response = client.chat(
        model=Config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": _route_prompt(query)},
        ],
        options={"temperature": 0.0},
    )
    return _parse_router_output(response["message"]["content"])


def _route_with_grok(query: str) -> dict | None:
    if not Config.GROK_API_KEY:
        return None

    url = f"{Config.GROK_BASE_URL.rstrip('/')}/chat/completions"
    response = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {Config.GROK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": Config.GROK_MODEL,
            "messages": [
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": _route_prompt(query)},
            ],
            "temperature": 0.0,
            "stream": False,
        },
        timeout=Config.GROK_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return _parse_router_output(data["choices"][0]["message"]["content"])


def route_query(query: str) -> dict:
    """
    Return {"route": "metadata"|"retrieval", "intent": "...", "confidence": ...}
    Uses the configured LLM when available and falls back to regex heuristics.
    """
    query = _normalize(query)
    if not query:
        return {"route": "retrieval", "intent": "", "confidence": 0.0}

    provider = Config.LLM_PROVIDER
    if provider not in {"auto", "ollama", "bedrock", "grok"}:
        provider = "auto"

    routers = []
    if provider == "bedrock":
        routers = [_route_with_bedrock]
    elif provider == "ollama":
        routers = [_route_with_ollama]
    elif provider == "grok":
        routers = [_route_with_grok]
    else:
        if Config.BEDROCK_MODEL_ID:
            routers.append(_route_with_bedrock)
        routers.append(_route_with_ollama)
        routers.append(_route_with_grok)

    for router in routers:
        try:
            result = router(query)
            if result:
                logger.info(f"Router classified query as {result}")
                return result
        except Exception as e:
            logger.warning(f"Router attempt failed: {e}")

    logger.info("Router fell back to retrieval")
    return {"route": "retrieval", "intent": "", "confidence": 0.0}


def is_metadata_candidate(query: str) -> bool:
    text = _normalize(query).lower()
    if not text:
        return False
    return any(hint in text for hint in _METADATA_HINTS)
