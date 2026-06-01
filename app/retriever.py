from collections import defaultdict

import chromadb
from rank_bm25 import BM25Okapi

from config import Config
from app.embedder import embed
from app.logger import get_logger

logger = get_logger(__name__)

_collection  = None
_bm25_index  = None   # BM25Okapi instance
_bm25_corpus = None   # list of passage dicts parallel to BM25 index
_reranker    = None   # CrossEncoder | False (False = tried and failed, don't retry)


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def get_collection():
    global _collection
    if _collection is None:
        logger.info(f"Connecting to ChromaDB at {Config.CHROMA_PATH}")
        try:
            client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
            _collection = client.get_collection(Config.CHROMA_COLLECTION)
            logger.info(
                f"ChromaDB collection '{Config.CHROMA_COLLECTION}' loaded "
                f"— {_collection.count()} chunks"
            )
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}")
            raise RuntimeError(
                "Vector database not found. Please run ingest.py first."
            ) from e
    return _collection


# ── BM25 index (built once from entire ChromaDB collection) ──────────────────

def get_bm25_index():
    """
    Loads the full ChromaDB collection into memory once and builds a BM25
    index over the document text.

    BM25 is keyword-frequency based — it does not understand semantics but
    it finds chunks whose tokens match the query even when embedding distance
    is poor (vocabulary mismatch). Critical for queries like
    "social tipping points" where the relevant section uses narrative language.
    """
    global _bm25_index, _bm25_corpus
    if _bm25_index is None:
        collection = get_collection()
        total = collection.count()

        logger.info(f"Building BM25 index over {total} chunks...")
        all_data = collection.get(
            include=["documents", "metadatas"],
            limit=total
        )

        _bm25_corpus = [
            {
                "document":       doc,
                "section_number": meta.get("section_number", ""),
                "section_title":  meta.get("section_title",  ""),
                # BM25 passages intentionally have no "distance" key —
                # handled by .get("distance", None) in stage 6
            }
            for doc, meta in zip(all_data["documents"], all_data["metadatas"])
        ]

        tokenized    = [p["document"].lower().split() for p in _bm25_corpus]
        _bm25_index  = BM25Okapi(tokenized)
        logger.info("BM25 index built successfully")

    return _bm25_index, _bm25_corpus


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _rrf_fuse(
    dense_passages:  list[dict],
    sparse_passages: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

        score(chunk) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)

    A chunk appearing in both lists gets a boosted combined score.
    A chunk appearing in only one list still contributes.
    k=60 is the standard value — no tuning required.
    """
    scores: dict[str, float] = defaultdict(float)
    index:  dict[str, dict]  = {}

    for rank, p in enumerate(dense_passages, start=1):
        key           = p["document"]
        scores[key]  += 1.0 / (k + rank)
        index[key]    = p

    for rank, p in enumerate(sparse_passages, start=1):
        key           = p["document"]
        scores[key]  += 1.0 / (k + rank)
        if key not in index:
            index[key] = p

    return sorted(index.values(), key=lambda p: scores[p["document"]], reverse=True)


# ── Cross-encoder reranker (optional, loaded lazily) ─────────────────────────

def _get_reranker():
    """
    Lazily load the cross-encoder reranker.

    Unlike the bi-encoder (MiniLM) which encodes query and document
    independently, the cross-encoder receives both concatenated and can model
    token-level interactions — allowing it to recognise that
    "horses-to-cars transition" is relevant to "social tipping points"
    even when surface vocabulary differs.

    Sentinel pattern:
      None  → not loaded yet
      False → tried to load and failed — do not retry
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(
                f"Loading cross-encoder reranker ({Config.RERANKER_MODEL})..."
            )
            _reranker = CrossEncoder(Config.RERANKER_MODEL)
            logger.info("Cross-encoder loaded successfully")
        except Exception as e:
            logger.warning(
                f"Cross-encoder unavailable ({e}) — rerank step will be skipped"
            )
            _reranker = False   # sentinel: tried and failed — don't retry
    return _reranker if _reranker is not False else None


def _rerank(query: str, passages: list[dict], top_n: int) -> list[dict]:
    """
    Score each (query, passage) pair jointly with the cross-encoder and
    return the top_n highest-scoring passages.

    If the cross-encoder is unavailable or there are fewer passages than
    top_n, returns passages[:top_n] unchanged.
    """
    reranker = _get_reranker()
    if reranker is None or len(passages) <= top_n:
        return passages[:top_n]

    pairs  = [(query, p["document"]) for p in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, passages), key=lambda x: x[0], reverse=True)

    reranked = [p for _, p in ranked[:top_n]]
    logger.info(
        f"Cross-encoder reranked {len(passages)} → {len(reranked)} passages "
        f"(top score: {ranked[0][0]:.3f})"
    )
    return reranked


# ── Section-title keyword boost ───────────────────────────────────────────────

def _section_keyword_boost(query: str, passages: list[dict]) -> list[dict]:
    """
    Promotes passages whose section_title overlaps with query tokens.
    Runs before the cross-encoder (O(n), microseconds) to ensure
    title-matching passages reach the reranking stage.

    Example: query "social tipping points" → "Tipping Points - Social"
    overlaps on "tipping", "points", "social" → all §12 chunks promoted.
    """
    query_tokens = set(query.lower().split())
    boosted, rest = [], []

    for p in passages:
        title_tokens = set(p.get("section_title", "").lower().split())
        if query_tokens & title_tokens:
            boosted.append(p)
        else:
            rest.append(p)

    if boosted:
        logger.info(
            f"Section-title boost: promoted {len(boosted)} passages "
            f"({[p['section_title'] for p in boosted]})"
        )

    return boosted + rest


# ── Worker warm-up ────────────────────────────────────────────────────────────

def warm_up():
    """
    Pre-load all expensive singletons at Celery worker startup.

    Called from app/tasks.py via the worker_ready signal. Prevents
    first-request latency spikes from lazy-loading the BM25 index
    (~50ms) and cross-encoder (~200ms + possible HuggingFace download).
    """
    try:
        get_bm25_index()
        logger.info("BM25 index warm-up complete")
    except Exception as e:
        logger.warning(f"BM25 warm-up failed: {e}")

    try:
        _get_reranker()
        logger.info("Cross-encoder warm-up complete")
    except Exception as e:
        logger.warning(f"Cross-encoder warm-up failed: {e}")


# ── Public entry point ────────────────────────────────────────────────────────

def retrieve(query: str) -> list[dict]:
    """
    6-stage hybrid retrieval pipeline.

    Stage 1 — Dense retrieval     ChromaDB cosine search (TOP_K × 4 candidates)
    Stage 2 — Sparse retrieval    BM25 keyword search    (same candidate budget)
    Stage 3 — RRF fusion          Merge both ranked lists
    Stage 4 — Section-title boost Promote title-matching passages
    Stage 5 — Cross-encoder       Rerank fused candidates → final TOP_K
    Stage 6 — Distance filter     Soft threshold (fallback: return all if empty)
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    log_query = query[:80] + "..." if len(query) > 80 else query
    logger.info(f"Retrieving chunks for query: '{log_query}'")

    # ── Stage 1: Dense retrieval (ChromaDB) ───────────────────────────────
    try:
        query_vector = embed(query)
    except RuntimeError:
        raise

    try:
        collection   = get_collection()
        n_candidates = min(Config.TOP_K * 4, collection.count())

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"ChromaDB query failed: {e}")
        raise RuntimeError(f"Vector search failed: {e}") from e

    dense_passages = [
        {
            "document":       doc,
            "section_number": meta.get("section_number", ""),
            "section_title":  meta.get("section_title",  ""),
            "distance":       dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
    logger.info(f"Dense retrieval: {len(dense_passages)} candidates")

    # ── Stage 2: Sparse retrieval (BM25) ──────────────────────────────────
    try:
        bm25, corpus    = get_bm25_index()
        bm25_scores     = bm25.get_scores(query.lower().split())

        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:n_candidates]

        sparse_passages = [
            {**corpus[i], "bm25_score": bm25_scores[i]}
            for i in top_indices
            if bm25_scores[i] > 0
        ]
        logger.info(f"BM25 retrieval: {len(sparse_passages)} candidates")

    except Exception as e:
        # BM25 failure must never break the pipeline
        logger.warning(f"BM25 retrieval failed ({e}) — using dense results only")
        sparse_passages = []

    # ── Stage 3: RRF fusion ────────────────────────────────────────────────
    fused = _rrf_fuse(dense_passages, sparse_passages) if sparse_passages else dense_passages
    logger.info(f"After RRF fusion: {len(fused)} unique candidates")

    # ── Stage 4: Section-title keyword boost ──────────────────────────────
    fused = _section_keyword_boost(query, fused)

    # ── Stage 5: Cross-encoder rerank → final TOP_K ───────────────────────
    passages = _rerank(query, fused, top_n=Config.TOP_K)

    # ── Stage 6: Distance-threshold filter (soft) ─────────────────────────
    # BM25-only results have no "distance" key — they default to None and
    # are treated as passing the filter (included unconditionally).
    filtered = [
        p for p in passages
        if p.get("distance") is None or p["distance"] < Config.DISTANCE_THRESHOLD
    ]

    if not filtered:
        logger.warning(
            f"No chunks below threshold {Config.DISTANCE_THRESHOLD}. "
            f"Using all {len(passages)} reranked results as fallback."
        )
        filtered = passages

    # Log distance range — only for chunks that have a real cosine distance
    real_distances = [p["distance"] for p in filtered if p.get("distance") is not None]
    bm25_only_count = len(filtered) - len(real_distances)

    if real_distances:
        dist_info = (
            f"distance range: {min(real_distances):.3f}–{max(real_distances):.3f}"
        )
        if bm25_only_count:
            dist_info += f" ({bm25_only_count} BM25-only)"
    else:
        dist_info = "BM25-only results (no cosine distance)"

    logger.info(f"Retrieved {len(filtered)} chunks — {dist_info}")

    return filtered