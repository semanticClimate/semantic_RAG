import chromadb
from config import Config
from app.embedder import embed
from app.logger import get_logger

logger = get_logger(__name__)

_collection = None


def get_collection():
    global _collection

    if _collection is None:
        logger.info(f"Connecting to ChromaDB at {Config.CHROMA_PATH}")

        try:
            chroma_client = chromadb.PersistentClient(
                path=Config.CHROMA_PATH
            )

            _collection = chroma_client.get_collection(
                Config.CHROMA_COLLECTION
            )

            count = _collection.count()

            logger.info(
                f"ChromaDB collection '{Config.CHROMA_COLLECTION}' loaded "
                f"— {count} chunks"
            )

        except Exception as e:
            logger.error(
                f"Failed to load ChromaDB collection "
                f"'{Config.CHROMA_COLLECTION}': {e}"
            )
            raise RuntimeError(
                "Vector database not found or invalid. "
                "Please run ingest.py first."
            ) from e

    return _collection


def _translate_to_english(query: str, language: str) -> str:
    """
    Translate a non-English query to English before embedding.
    This is necessary because the embedding model indexes English passages,
    so non-English queries produce mismatched vectors and miss relevant chunks.
    Falls back to the original query if translation fails.
    """
    if language == "English":
        return query

    try:
        import ollama
        client_kwargs = {"host": Config.OLLAMA_BASE_URL}
        if Config.OLLAMA_API_KEY:
            client_kwargs["headers"] = {"Authorization": f"Bearer {Config.OLLAMA_API_KEY}"}
        client = ollama.Client(**client_kwargs)
        response = client.chat(
            model=Config.OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the following text to English. "
                        "Output ONLY the English translation — no explanations, "
                        "no notes, no punctuation changes."
                    ),
                },
                {"role": "user", "content": query},
            ],
            options={"temperature": 0.0},
        )
        translated = response["message"]["content"].strip()
        logger.info(f"Query translated ({language} → English): '{translated[:80]}'")
        return translated

    except Exception as e:
        logger.warning(
            f"Translation failed for language='{language}': {e} "
            f"— embedding original query (retrieval may be weaker)"
        )
        return query  # non-fatal fallback


def retrieve(query: str, language: str = "English") -> list[dict]:
    """
    Two-phase retrieval strategy:

    Phase 1 — Search book chunks only.
      - If strong book results exist (any passage below DISTANCE_THRESHOLD),
        also run Phase 2 in parallel so the encyclopedia can supplement.
      - If NO book results pass the threshold, Phase 2 runs as the sole source.

    Phase 2 — Search encyclopedia chunks only.
      - Always runs alongside Phase 1 when book results are weak/absent.
      - Results are merged with book results, deduplicated, and re-sorted
        by distance so the LLM always gets the best-matching context first.

    Why two separate queries instead of one combined query:
      - A single TOP_K query across all chunks lets whichever source type
        happens to have more chunks (usually the book) crowd out the other.
      - Filtering per source_type gives each source a fair TOP_K budget,
        so encyclopedia entries are never starved by book chunks.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info(
        f"Retrieving chunks for query: '{query[:80]}...'"
        if len(query) > 80
        else f"Retrieving chunks for query: '{query}'"
    )

    # Translate to English before embedding — the corpus is in English,
    # so non-English queries must be translated first to get matching vectors.
    english_query = _translate_to_english(query, language)

    # Generate query embedding once — reused for both phases
    try:
        query_vector = embed(english_query)
    except RuntimeError as e:
        logger.error(f"Embedding failed during retrieval: {e}")
        raise

    # ── Phase 1: book chunks ──────────────────────────────────────────────────
    book_passages = _query_source(
        query_vector,
        source_type="book",
        n_results=Config.TOP_K,
    )

    book_hits = [p for p in book_passages if p["distance"] < Config.DISTANCE_THRESHOLD]
    logger.info(f"Phase 1 (book): {len(book_hits)} hits below threshold out of {len(book_passages)} fetched")

    # ── Phase 2: encyclopedia chunks ─────────────────────────────────────────
    # Always run Phase 2 so the encyclopedia can supplement the book.
    # If book results are strong, enc results act as additional context.
    # If book results are empty, enc results are the primary answer source.
    enc_passages = _query_source(
        query_vector,
        source_type="encyclopedia",
        n_results=Config.TOP_K,
    )

    enc_hits = [p for p in enc_passages if p["distance"] < Config.DISTANCE_THRESHOLD]
    logger.info(f"Phase 2 (encyclopedia): {len(enc_hits)} hits below threshold out of {len(enc_passages)} fetched")

    # ── Merge & deduplicate ───────────────────────────────────────────────────
    # Combine both hit lists, remove any duplicate section+chunk combos,
    # and sort by distance (closest = most relevant first).
    combined = book_hits + enc_hits
    combined.sort(key=lambda p: p["distance"])

    # Deduplicate by (source_type, section_number, document text)
    seen: set[tuple] = set()
    passages: list[dict] = []
    for p in combined:
        key = (p["source_type"], p["section_number"], p["document"][:80])
        if key not in seen:
            seen.add(key)
            passages.append(p)

    # Cap total passages to avoid bloating the LLM context window
    max_passages = getattr(Config, "MAX_PASSAGES", Config.TOP_K * 2)
    passages = passages[:max_passages]

    if not passages:
        logger.warning(
            f"No chunks below threshold {Config.DISTANCE_THRESHOLD} in either "
            f"book or encyclopedia — returning empty (will trigger 'not found' response)."
        )
        return []
    logger.info("Final passages sent to LLM:")

    for i, p in enumerate(passages):
        logger.info(
            f"{i+1}. "
            f"{p['source_type']} "
            f"{p['distance']:.3f} "
            f"{p.get('section_title', '')}"
        )

    distances = [p["distance"] for p in passages]
    book_count = sum(1 for p in passages if p["source_type"] == "book")
    enc_count  = sum(1 for p in passages if p["source_type"] == "encyclopedia")

    logger.info(
        f"Retrieved {len(passages)} passages total "
        f"(book: {book_count}, encyclopedia: {enc_count}) — "
        f"distance range: {min(distances):.3f}–{max(distances):.3f}"
    )

    return passages


def _query_source(
    query_vector: list[float],
    source_type: str,
    n_results: int,
) -> list[dict]:
    """
    Query ChromaDB filtered to a single source_type.
    Returns raw passage dicts (no threshold filtering — caller decides).
    Returns [] gracefully if the collection has no docs of that type.
    """
    try:
        collection = get_collection()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where={"source_type": {"$eq": source_type}},
            include=["documents", "metadatas", "distances"],
        )

    except RuntimeError:
        raise
    except Exception as e:
        # If the filtered query fails (e.g. no docs of that type exist yet),
        # log and return empty rather than crashing the whole request.
        logger.warning(f"ChromaDB query failed for source_type='{source_type}': {e}")
        return []

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        _build_passage(doc, meta, dist)
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]


def _build_passage(doc: str, meta: dict, dist: float) -> dict:
    """
    Build a passage dict from a ChromaDB result row.
    Pulls all rich metadata stored by ingest.py so callers
    (the Flask route, the source-label builder, etc.) have
    everything they need without touching ChromaDB again.
    """
    source_type = meta.get("source_type", "book")  # "book" | "encyclopedia"

    passage = {
        "document":       doc,
        "source_type":    source_type,
        "source_path":    meta.get("source_path", ""),
        "section_number": meta.get("section_number", ""),
        "section_title":  meta.get("section_title", ""),
        "distance":       dist,
    }

    if source_type == "book":
        passage["chapter_number"] = int(meta.get("chapter_number", 0))
        passage["chapter_title"]  = meta.get("chapter_title", "")
    else:
        # encyclopedia
        passage["term"] = meta.get("term", meta.get("section_title", ""))

    return passage
