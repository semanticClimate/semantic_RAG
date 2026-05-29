import ollama
import httpx
from config import Config
from app.logger import get_logger

logger = get_logger(__name__)


def build_system_prompt(
    passages: list[dict],
    query: str,
    language: str = "the same language as the user",
) -> str:
    context_block = "\n\n".join(p["document"] for p in passages)
    return f"""You are the Climate Academy study assistant.
Use the provided book passages as your primary source and answer in clear student-friendly language.

Rules:
1. Ground your answer in the passages and do not invent facts.
2. If passages are partial, give the best supported answer and state what is uncertain.
3. If the answer is truly missing, say: "I could not find that in the Climate Academy book." and ask one short follow-up question.
4. Keep the response concise but meaningful (4-8 sentences for normal questions).
5. Reply in {language}.

--- BOOK PASSAGES ---
{context_block}
--- END OF PASSAGES ---

User question: {query}

Answer with:
- Short direct answer
- 2-4 key points
- Optional one-line clarification if needed
"""


def generate(
    passages: list[dict],
    history: list[dict],
    user_message: str,
    language: str = "English",
) -> str:
    if not passages:
        logger.warning("generate() called with empty passages list")

    system_prompt = build_system_prompt(passages, user_message, language)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    provider = Config.LLM_PROVIDER
    if provider not in {"auto", "ollama", "grok"}:
        logger.warning(f"Unknown LLM_PROVIDER '{provider}', falling back to auto")
        provider = "auto"

    if provider == "grok":
        return _generate_with_grok(messages)

    if provider == "ollama":
        return _generate_with_ollama(messages, len(passages), len(history))

    try:
        return _generate_with_ollama(messages, len(passages), len(history))
    except RuntimeError as e:
        if not Config.GROK_API_KEY:
            raise
        logger.warning(f"Ollama unavailable, falling back to Grok: {e}")
        return _generate_with_grok(messages)


def _generate_with_ollama(messages: list[dict], passage_count: int, history_count: int) -> str:
    logger.info(
        f"Calling Ollama model '{Config.OLLAMA_MODEL}' at {Config.OLLAMA_BASE_URL} - "
        f"{passage_count} passages, {history_count} history messages"
    )

    try:
        client = ollama.Client(host=Config.OLLAMA_BASE_URL)
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
            "Ollama request timed out. The model may be overloaded - please try again."
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
