import chromadb
from pathlib import Path
from config import Config
from html_sectioning import parse_html_path_to_chunks, detect_source_type, load_html_file


def run_ingestion():
    print("Starting ingestion...")

    # chunk_mode "auto" is now the recommended default:
    #   - book records   → word-level sliding-window chunks
    #   - encyclopedia   → sentence-aware chunks
    # Override via Config.CHUNK_MODE = "default" | "encyclopedia" | "auto"
    chunk_mode = Config.CHUNK_MODE if Config.CHUNK_MODE in {"default", "encyclopedia", "auto"} else "auto"
    print(f"Chunk mode: {chunk_mode}")

    source_paths = Config.SOURCE_HTML_PATHS or [Config.SOURCE_HTML_PATH]
    source_paths = [Path(p) for p in source_paths]
    print(f"Source HTML files: {[str(p) for p in source_paths]}")

    sourced_chunks = []
    for source_path in source_paths:
        # Auto-detect what kind of source this is so we can report it clearly
        html_preview = load_html_file(source_path)
        detected_type = detect_source_type(html_preview)
        print(f"  Detected source type for {source_path.name}: {detected_type}")

        source_chunks = parse_html_path_to_chunks(
            path=source_path,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            chunk_mode=chunk_mode,
        )
        sourced_chunks.extend((str(source_path), c) for c in source_chunks)
        print(f"  Parsed {len(source_chunks)} chunks from {source_path.name}")

        # Summary breakdown
        book_chunks = sum(1 for _, c in sourced_chunks if c.source_type == "book")
        enc_chunks  = sum(1 for _, c in sourced_chunks if c.source_type == "encyclopedia")
    print(f"\nTotal chunks: {len(sourced_chunks)}  (book: {book_chunks}, encyclopedia: {enc_chunks})")

    from app.embedder import embed

    chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)

    # Wipe and recreate collection for clean re-ingestion
    try:
        chroma_client.delete_collection(Config.CHROMA_COLLECTION)
        print(f"Dropped existing collection '{Config.CHROMA_COLLECTION}'")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=Config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    for i in range(0, len(sourced_chunks), batch_size):
        batch = sourced_chunks[i: i + batch_size]

        ids = [
            f"{c.source_type}__{c.section_number}__{c.chunk_index}__{i + j}"
            for j, (src, c) in enumerate(batch)
        ]
        documents  = [c.document  for _, c in batch]
        embeddings = [embed(c.document) for _, c in batch]

        metadatas = []
        for src, c in batch:
            meta = {
                # ── common fields ──────────────────────────────────────────
                "source_path":    src,
                "source_type":    c.source_type,      # "book" | "encyclopedia"
                "section_number": c.section_number,
                "section_title":  c.section_title,
                "chunk_index":    c.chunk_index,
            }

            if c.source_type == "book":
                # ── book-specific ──────────────────────────────────────────
                meta["chapter_number"] = c.chapter_number   # int 0-16
                meta["chapter_title"]  = c.chapter_title    # e.g. "Spaceship Earth"
            else:
                # ── encyclopedia-specific ──────────────────────────────────
                # section_number holds the term slug; section_title holds the
                # display name — both already set correctly in SectionRecord.
                # We add a convenience field for quick filtering.
                meta["chapter_number"] = 0
                meta["chapter_title"]  = ""
                meta["term"]           = c.section_title    # e.g. "carbon cycle"

            metadatas.append(meta)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  Indexed chunks {i + 1}–{min(i + batch_size, len(sourced_chunks))}")

    print(f"\nIngestion complete — {collection.count()} chunks in ChromaDB.")


if __name__ == "__main__":
    run_ingestion()