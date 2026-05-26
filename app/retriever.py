import chromadb
from config import Config
from app.embedder import embed
from app.logger import get_logger

logger = get_logger(__name__)

_client = None
_collections = {}


def _get_client():
    global _client

    if _client is None:
        logger.info(f"Connecting to ChromaDB at {Config.CHROMA_PATH}")
        try:
            _client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB at '{Config.CHROMA_PATH}': {e}")
            raise RuntimeError(
                "Vector database not found or invalid. "
                "Please run ingest.py first."
            ) from e

    return _client


def _get_collection(name: str, required: bool):
    """
    Lazily load a Chroma collection.
    - required=True: raises if missing (keeps existing system behavior)
    - required=False: returns None if missing (optional extra sources)
    """
    if name in _collections:
        return _collections[name]

    client = _get_client()
    try:
        col = client.get_collection(name)
        _collections[name] = col
        try:
            count = col.count()
            logger.info(f"ChromaDB collection '{name}' loaded — {count} chunks")
        except Exception:
            logger.info(f"ChromaDB collection '{name}' loaded")
        return col
    except Exception as e:
        msg = f"Failed to load ChromaDB collection '{name}': {e}"
        if required:
            logger.error(msg)
            raise RuntimeError(
                "Vector database not found or invalid. "
                "Please run ingest.py first."
            ) from e
        logger.warning(msg)
        return None


def retrieve(query: str) -> list[dict]:

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    logger.info(
        f"Retrieving chunks for query: "
        f"'{query[:80]}...'"
        if len(query) > 80
        else f"Retrieving chunks for query: '{query}'"
    )

    # Generate embedding
    try:
        query_vector = embed(query)
    except RuntimeError as e:
        logger.error(f"Embedding failed during retrieval: {e}")
        raise

    # Always query the existing book collection (required)
    book_col = _get_collection(Config.CHROMA_COLLECTION, required=True)
    collections = [(Config.CHROMA_COLLECTION, book_col)]

    # Optionally query the climate encyclopedia collection if present
    extra = _get_collection(Config.CHROMA_COLLECTION_CLIMATE_HTML, required=False)
    if extra is not None:
        collections.append((Config.CHROMA_COLLECTION_CLIMATE_HTML, extra))

    merged = []

    for col_name, col in collections:
        try:
            results = col.query(
                query_embeddings=[query_vector],
                n_results=Config.TOP_K,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed for '{col_name}': {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            merged.append(
                {
                    "document": doc,
                    "section_number": meta.get("section_number", ""),
                    "section_title": meta.get("section_title", ""),
                    "distance": dist,
                    "source": meta.get("source", col_name),
                }
            )

    # Sort by distance (lower is better for cosine distance)
    merged.sort(key=lambda p: p.get("distance", 1e9))

    passages = [p for p in merged if p["distance"] < Config.DISTANCE_THRESHOLD]

    # Fallback if no chunks under threshold: return best overall results
    if not passages:
        logger.warning(
            f"No chunks below threshold {Config.DISTANCE_THRESHOLD}. "
            f"Using top results as fallback."
        )
        passages = merged

    passages = passages[: Config.TOP_K]

    if passages:
        distances = [p["distance"] for p in passages]
        logger.info(
            f"Retrieved {len(passages)} chunks — "
            f"distance range: {min(distances):.3f}–{max(distances):.3f}"
        )
    else:
        logger.info("Retrieved 0 chunks")

    return passages
