import chromadb
from config import Config
from app.embedder import embed
from app.logger import get_logger
from app.book_facts import build_fact_passages, is_chapter_summary_query
from app.query_router import route_query
from app.translation import translate_to_english

logger = get_logger(__name__)

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    logger.info(f"Connecting to ChromaDB at {Config.CHROMA_PATH}")
    try:
        chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        _collection = chroma_client.get_collection(Config.CHROMA_COLLECTION)
        logger.info(
            f"ChromaDB collection '{Config.CHROMA_COLLECTION}' loaded — {_collection.count()} chunks"
        )
        return _collection
    except Exception as e:
        logger.error(f"Failed to load ChromaDB collection '{Config.CHROMA_COLLECTION}': {e}")
        raise RuntimeError("Vector database not found or invalid. Please run ingest.py first.") from e


def _query_source(query_vector, source_type: str, n_results: int) -> list[dict]:
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where={"source_type": source_type},
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    passages = []
    for doc, meta, dist in zip(docs, metas, dists):
        passage = dict(meta)
        passage["document"] = doc
        passage["distance"] = dist
        passage["source_type"] = meta.get("source_type", source_type)
        passages.append(passage)
    return passages


def retrieve_chapter_passages(query_vector, chapter_number: int) -> list[dict]:
    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=Config.TOP_K,
        where={
            "$and": [
                {"source_type": "book"},
                {"chapter_number": chapter_number},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return [
        {**meta, "document": doc, "distance": dist, "source_type": meta.get("source_type", "book")}
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def retrieve(query: str, language: str = "English") -> list[dict]:
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info(f"Retrieving chunks for query: '{query[:80]}...'" if len(query) > 80 else f"Retrieving chunks for query: '{query}'")

    english_query = translate_to_english(query, language)
    route = route_query(english_query)
    if route.get("route") == "metadata":
        fact_passages = build_fact_passages(english_query, intent=route.get("intent") or None)
        if fact_passages:
            logger.info(f"Using metadata shortcut for query: '{query[:80]}'")
            return fact_passages

    english_query = translate_to_english(query, language)
    try:
        query_vector = embed(english_query)
    except RuntimeError as e:
        logger.error(f"Embedding failed during retrieval: {e}")
        raise

    chapter_number = is_chapter_summary_query(english_query)
    if chapter_number is not None:
        chapter_passages = retrieve_chapter_passages(query_vector, chapter_number)
        if chapter_passages:
            logger.info(f"Using chapter-scoped retrieval for Chapter {chapter_number}")
            return chapter_passages

    book_passages = _query_source(query_vector, source_type="book", n_results=Config.TOP_K)
    enc_passages = _query_source(query_vector, source_type="encyclopedia", n_results=Config.TOP_K)

    combined = book_passages + enc_passages
    combined.sort(key=lambda p: p.get("distance", 999.0))

    seen = set()
    passages = []
    for p in combined:
        key = (p.get("source_type"), p.get("section_number"), p.get("document", "")[:80])
        if key in seen:
            continue
        seen.add(key)
        passages.append(p)

    max_passages = getattr(Config, "MAX_PASSAGES", Config.TOP_K * 2)
    passages = passages[:max_passages]

    if not passages:
        return []

    logger.info("Final passages sent to LLM:")
    for i, p in enumerate(passages, start=1):
        logger.info(f"{i}. {p.get('source_type')} {p.get('distance', 0.0):.3f} {p.get('section_title', '')}")

    return passages
