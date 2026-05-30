import sys
import ollama
import httpx
from groq import Groq
from groq import APIConnectionError as GroqConnectionError
from groq import APIStatusError as GroqStatusError
from config import Config
from app.logger import get_logger

logger = get_logger(__name__)

# Initialise Groq client once (no-op if key is empty)
_groq_client = Groq(api_key=Config.GROQ_API_KEY) if Config.GROQ_API_KEY else None

# Keywords in ollama exception messages that mean "can't reach the server"
_OLLAMA_CONNECT_PHRASES = (
    "failed to connect",
    "connection refused",
    "connect error",
    "ollama is downloaded",   # the exact phrase from the library's own message
    "name or service not known",
    "no route to host",
)


def _is_ollama_connection_error(exc: Exception) -> bool:
    return any(phrase in str(exc).lower() for phrase in _OLLAMA_CONNECT_PHRASES)


def build_system_prompt(passages: list[dict]) -> str:
    context_block = "\n\n".join(p["document"] for p in passages)
    return f"""You are a helpful assistant for the Climate Academy student book.
Answer questions strictly based on the provided passages below.
Do not use any outside knowledge. If the answer is not in the passages, say so explicitly.
Cite section numbers inline using the format §x.y.z wherever relevant.
Automatically detect and respond in the same language as the user (English, Hindi, or French).
Use bullet points where appropriate for clarity.

RETRIEVED PASSAGES:
{context_block}"""


# ── Ollama ────────────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict]) -> str:
    """Call local Ollama. Raises RuntimeError('ollama_unavailable') on connectivity issues."""
    logger.info(f"Calling Ollama model '{Config.OLLAMA_MODEL}'")
    try:
        response = ollama.chat(
            model=Config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.3}
        )
        answer = response["message"]["content"]
        logger.info(f"Ollama responded — {len(answer)} chars generated")
        return answer

    except ollama.ResponseError as e:
        logger.error(f"Ollama API error (status {e.status_code}): {e.error}")
        if e.status_code == 404:
            raise RuntimeError(
                f"Model '{Config.OLLAMA_MODEL}' not found. "
                f"Run: ollama pull {Config.OLLAMA_MODEL}"
            ) from e
        raise RuntimeError(f"Ollama API error: {e.error}") from e

    except httpx.ConnectError as e:
        logger.warning(f"Ollama ConnectError: {e}")
        raise RuntimeError("ollama_unavailable") from e

    except httpx.TimeoutException as e:
        logger.warning(f"Ollama timeout: {e}")
        raise RuntimeError("ollama_unavailable") from e

    except Exception as e:
        # The ollama library raises plain Exception for connection failures
        # (e.g. "Failed to connect to Ollama...") — catch those here too
        if _is_ollama_connection_error(e):
            logger.warning(f"Ollama unreachable (caught via message match): {e}")
            raise RuntimeError("ollama_unavailable") from e
        logger.error(f"Unexpected Ollama error: {e}")
        raise RuntimeError(f"ollama_unexpected: {e}") from e


# ── Groq ──────────────────────────────────────────────────────────────────────

def _call_groq(messages: list[dict]) -> str:
    """Call Groq's LLaMA API. Raises RuntimeError on any failure."""
    if _groq_client is None:
        raise RuntimeError(
            "Groq fallback is not configured. Set GROQ_API_KEY in your .env file."
        )

    logger.info(f"Calling Groq model '{Config.GROQ_MODEL}' (fallback)")
    try:
        response = _groq_client.chat.completions.create(
            model=Config.GROQ_MODEL,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        logger.info(f"Groq responded — {len(answer)} chars generated")
        return answer

    except GroqConnectionError as e:
        logger.error(f"Cannot connect to Groq API: {e}")
        raise RuntimeError("Cannot reach Groq API. Check your network.") from e

    except GroqStatusError as e:
        logger.error(f"Groq API error (status {e.status_code}): {e.message}")
        raise RuntimeError(f"Groq API error: {e.message}") from e

    except Exception as e:
        logger.error(f"Unexpected Groq error: {e}")
        raise RuntimeError(f"Groq call failed: {e}") from e


# ── Public entry point ────────────────────────────────────────────────────────

def generate(
    passages: list[dict],
    history: list[dict],
    user_message: str
) -> tuple[str, str]:
    """
    Generate a response, trying Ollama first and falling back to Groq.

    Returns:
        (answer, backend)  where backend is "ollama" or "groq".
    """
    if not passages:
        logger.warning("generate() called with empty passages list")

    system_prompt = build_system_prompt(passages)
    messages = [{"role": "system", "content": system_prompt}]
    messages += history
    messages.append({"role": "user", "content": user_message})

    logger.info(f"{len(passages)} passages, {len(history)} history messages")

    # 1. Try Ollama
    try:
        answer = _call_ollama(messages)
        return answer, "ollama"
    except RuntimeError as e:
        if "ollama_unavailable" not in str(e):
            raise  # config errors (404, etc.) bubble up immediately

    # 2. Ollama is down — print a visible warning and fall back to Groq
    _warn_fallback_to_groq()
    answer = _call_groq(messages)
    return answer, "groq"


def _warn_fallback_to_groq() -> None:
    """Print a bold yellow warning to the terminal (visible in Celery logs)."""
    YELLOW = "\033[1;33m"
    RESET  = "\033[0m"
    border = "=" * 60
    msg = (
        f"\n{YELLOW}{border}\n"
        f"  ⚠  OLLAMA UNAVAILABLE — switching to Groq fallback\n"
        f"     Model : {Config.GROQ_MODEL}\n"
        f"     Tip   : run 'sudo systemctl start ollama' to restore local inference\n"
        f"{border}{RESET}\n"
    )
    print(msg, file=sys.stderr, flush=True)
    logger.warning("Ollama unavailable — switched to Groq fallback")