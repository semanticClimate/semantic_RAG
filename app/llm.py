import httpx
from config import Config
from app.logger import get_logger

logger = get_logger(__name__)


def _format_passage(p: dict) -> str:
    """
    Render a single passage with a clear source header so the LLM knows
    whether it is reading from the book or the encyclopedia.
    """
    source_type = p.get("source_type", "book")

    if source_type == "book":
        ch_num   = p.get("chapter_number", "")
        ch_title = p.get("chapter_title", "") or p.get("section_title", "")
        header   = f"[BOOK — Ch.{ch_num}: {ch_title}]" if ch_num else f"[BOOK — {ch_title}]"
    else:
        term   = p.get("term", "") or p.get("section_title", "")
        header = f"[ENCYCLOPEDIA — {term}]"

    return f"{header}\n{p['document']}"


def build_system_prompt(
    passages: list[dict],
    language: str = "the same language as the user",
) -> str:
    context_block = "\n\n".join(_format_passage(p) for p in passages)

    has_book = any(p.get("source_type") == "book" for p in passages)
    has_enc  = any(p.get("source_type") == "encyclopedia" for p in passages)

    if has_book and has_enc:
        source_desc = "the Climate Academy book and the Climate Academy Encyclopedia"
    elif has_enc:
        source_desc = "the Climate Academy Encyclopedia"
    else:
        source_desc = "the Climate Academy book"

    return f"""You are the Climate Academy study assistant. You answer STRICTLY and ONLY from the passages provided below.

The passages come from {source_desc}. Each passage is labelled [BOOK — ...] or [ENCYCLOPEDIA — ...] so you know its origin.

ABSOLUTE RULES — never break these:

1. Your ONLY source of information is the PASSAGES block below.
2. You MUST NOT use any of your pre-trained / world knowledge — not even to add a sentence, clarify, or supplement.
3. You MUST NOT infer, extrapolate, or assume facts that are not explicitly stated in the passages.
4. You MUST NOT guess.
5. Use BOTH book and encyclopedia passages if they are relevant — they are complementary sources.
6. If the answer is not explicitly and clearly present in the passages, you MUST respond with exactly:
   "I could not find that in the Climate Academy materials."
   Do NOT attempt to answer even partially from memory.
7. If only part of the answer is in the passages, answer only that part and say the rest was not found in the materials.
8. Respond in {language}.

--- PASSAGES BEGIN ---
{context_block}
--- PASSAGES END ---

Answer the user's question using ONLY the passages above. If the answer is not there, say so."""


def generate(
    passages: list[dict],
    history: list[dict],
    user_message: str,
    language: str = "English",
) -> str:
    if not passages:
        return "I could not find that in the Climate Academy materials."

    system_prompt = build_system_prompt(passages, language)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    provider = Config.LLM_PROVIDER
    if provider not in {"auto", "ollama", "bedrock", "grok"}:
        logger.warning(f"Unknown LLM_PROVIDER '{provider}', falling back to auto")
        provider = "auto"

    if provider == "bedrock":
        return _generate_with_bedrock(system_prompt, [], user_message)

    if provider == "auto" and Config.BEDROCK_MODEL_ID:
        try:
            return _generate_with_bedrock(system_prompt, [], user_message)
        except RuntimeError as e:
            logger.warning(f"Bedrock unavailable, falling back to Ollama: {e}")

    if provider == "grok":
        return _generate_with_grok(messages)

    # Ollama: system prompt + current message only — no history — to keep tokens low.
    # Grok keeps the full messages list (with history) built above.
    ollama_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    if provider == "ollama":
        return _generate_with_ollama(ollama_messages, len(passages), len(history))

    try:
        return _generate_with_ollama(ollama_messages, len(passages), len(history))
    except RuntimeError as e:
        if not Config.GROK_API_KEY:
            raise
        logger.warning(f"Ollama unavailable, falling back to Grok: {e}")
        return _generate_with_grok(messages)


def _generate_with_bedrock(system_prompt: str, history: list[dict], user_message: str) -> str:
    import boto3

    logger.info(
        f"Calling Bedrock model '{Config.BEDROCK_MODEL_ID}' in region {Config.AWS_REGION}"
    )

    client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)

    try:
        bedrock_messages = []
        for message in history:
            role = message.get("role")
            content = message.get("content", "")
            if role not in {"user", "assistant"}:
                continue
            bedrock_messages.append({
                "role": role,
                "content": [{"text": str(content)}],
            })

        bedrock_messages.append({
            "role": "user",
            "content": [{"text": str(user_message)}],
        })

        response = client.converse(
            modelId=Config.BEDROCK_MODEL_ID,
            messages=bedrock_messages,
            system=[{"text": system_prompt}],
            inferenceConfig={
                "temperature": Config.BEDROCK_TEMPERATURE,
                "maxTokens": Config.BEDROCK_MAX_TOKENS,
            },
        )
        content = response["output"]["message"]["content"]
        answer = "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
        if not answer:
            raise RuntimeError("Bedrock returned an empty response.")
        logger.info(f"Bedrock responded - {len(answer)} chars generated")
        return answer

    except client.exceptions.AccessDeniedException as e:
        logger.error(f"Bedrock access denied: {e}")
        raise RuntimeError("Bedrock access denied. Check IAM permissions and model access.") from e
    except client.exceptions.ValidationException as e:
        logger.error(f"Bedrock validation error: {e}")
        raise RuntimeError(f"Bedrock validation error: {e}") from e
    except client.exceptions.ThrottlingException as e:
        logger.error(f"Bedrock throttled: {e}")
        raise RuntimeError("Bedrock is busy right now. Please try again in a minute.") from e
    except Exception as e:
        logger.error(f"Unexpected error calling Bedrock: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e


def _generate_with_ollama(messages: list[dict], passage_count: int, history_count: int) -> str:
    import ollama
    logger.info(
        f"Calling Ollama model '{Config.OLLAMA_MODEL}' at {Config.OLLAMA_BASE_URL} - "
        f"{passage_count} passages, {history_count} history messages"
    )

    try:
        client_kwargs = {"host": Config.OLLAMA_BASE_URL}
        if Config.OLLAMA_API_KEY:
            client_kwargs["headers"] = {"Authorization": f"Bearer {Config.OLLAMA_API_KEY}"}
        client = ollama.Client(**client_kwargs)
        response = client.chat(
            model=Config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.3}
        )
        answer = response["message"]["content"]
        logger.info(f"Ollama responded - {len(answer)} chars generated")
        return answer

    except ollama.ResponseError as e:
        # Model not found, bad request, etc.
        logger.error(f"Ollama API error (status {e.status_code}): {e.error}")
        if e.status_code == 404:
            raise RuntimeError(
                f"Model '{Config.OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {Config.OLLAMA_MODEL}"
            ) from e
        if e.status_code == 524 or "timeout" in str(e.error).lower():
            raise RuntimeError(
                "AI is busy right now. Please try again in a minute."
            ) from e
        raise RuntimeError(f"Ollama API error: {e.error}") from e

    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama at {Config.OLLAMA_BASE_URL}: {e}")
        raise RuntimeError(
            "Cannot connect to Ollama. "
            "Ensure the SSH tunnel is active and Ollama is running on the GPU server."
        ) from e

    except httpx.TimeoutException as e:
        logger.error(f"Ollama request timed out: {e}")
        raise RuntimeError(
            "AI is busy right now. Please try again in a minute."
        ) from e

    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e


def _generate_with_grok(messages: list[dict]) -> str:
    if not Config.GROK_API_KEY:
        raise RuntimeError(
            "Grok API key is not configured. Set GROK_API_KEY or XAI_API_KEY."
        )

    url = f"{Config.GROK_BASE_URL.rstrip('/')}/chat/completions"
    logger.info(f"Calling Grok model '{Config.GROK_MODEL}' at {Config.GROK_BASE_URL}")

    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {Config.GROK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": Config.GROK_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "stream": False,
            },
            timeout=Config.GROK_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        logger.info(f"Grok responded - {len(answer)} chars generated")
        return answer

    except httpx.HTTPStatusError as e:
        logger.error(f"Grok API error (status {e.response.status_code}): {e.response.text}")
        raise RuntimeError(f"Grok API error: HTTP {e.response.status_code}") from e

    except (httpx.ConnectError, httpx.TimeoutException) as e:
        logger.error(f"Cannot connect to Grok at {Config.GROK_BASE_URL}: {e}")
        raise RuntimeError("Cannot connect to Grok. Please try again later.") from e

    except (KeyError, IndexError, ValueError) as e:
        logger.error(f"Unexpected Grok response shape: {e}")
        raise RuntimeError("Grok returned an unexpected response.") from e

    except Exception as e:
        logger.error(f"Unexpected error calling Grok: {e}")
        raise RuntimeError(f"LLM call failed: {e}") from e
