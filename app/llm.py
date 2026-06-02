import ollama
import httpx
from config import Config
from app.logger import get_logger

logger = get_logger(__name__)
_bedrock_client = None


def _format_passage(p: dict) -> str:
    """Render a passage with a source header so the LLM knows its origin."""
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
2. Do NOT use any pre-trained / world knowledge — not even to add a sentence or clarify.
3. Do NOT infer, extrapolate, or assume facts not explicitly stated in the passages.
4. Use BOTH book and encyclopedia passages if they are relevant — they are complementary sources.
5. If the answer is not explicitly present, respond with exactly:
   "I could not find that in the Climate Academy materials."
6. If only part of the answer is in the passages, answer only that part and say the rest was not found.
7. Respond in {language}.

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

    if Config.LLM_PROVIDER == "bedrock":
        return _generate_with_bedrock(system_prompt, history, user_message, len(passages))

    if Config.LLM_PROVIDER != "ollama":
        raise RuntimeError("Unsupported LLM_PROVIDER. Use 'ollama' or 'bedrock'.")

    logger.info(
        f"Calling Ollama model '{Config.OLLAMA_MODEL}' at {Config.OLLAMA_BASE_URL} - "
        f"{len(passages)} passages, {len(history)} history messages"
    )

    try:
        client = ollama.Client(host=Config.OLLAMA_BASE_URL)
        response = client.chat(
            model=Config.OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.3},
        )
        answer = response["message"]["content"]
        logger.info(f"Ollama responded - {len(answer)} chars generated")
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


def get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        logger.info(f"Creating Bedrock Runtime client in {Config.AWS_REGION}")
        import boto3

        _bedrock_client = boto3.client("bedrock-runtime", region_name=Config.AWS_REGION)
    return _bedrock_client


def _to_bedrock_messages(history: list[dict], user_message: str) -> list[dict]:
    bedrock_messages = []
    for message in history:
        role = message.get("role")
        content = message.get("content", "")
        if role not in {"user", "assistant"} or not content:
            continue
        bedrock_messages.append({"role": role, "content": [{"text": content}]})

    bedrock_messages.append({"role": "user", "content": [{"text": user_message}]})
    return bedrock_messages


def _generate_with_bedrock(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    passage_count: int,
) -> str:
    logger.info(
        f"Calling Bedrock model '{Config.BEDROCK_CHAT_MODEL}' in {Config.AWS_REGION} - "
        f"{passage_count} passages, {len(history)} history messages"
    )

    try:
        client = get_bedrock_client()
        response = client.converse(
            modelId=Config.BEDROCK_CHAT_MODEL,
            system=[{"text": system_prompt}],
            messages=_to_bedrock_messages(history, user_message),
            inferenceConfig={
                "maxTokens": Config.BEDROCK_MAX_TOKENS,
                "temperature": Config.BEDROCK_TEMPERATURE,
            },
        )
        answer = response["output"]["message"]["content"][0]["text"]
        logger.info(f"Bedrock responded - {len(answer)} chars generated")
        return answer
    except Exception as e:
        logger.error(f"Bedrock LLM call failed: {e}")
        raise RuntimeError(f"Bedrock LLM call failed: {e}") from e
