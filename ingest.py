import chromadb
from pathlib import Path
from config import Config
from html_sectioning import parse_html_path_to_chunks

def run_ingestion():
    print("Starting ingestion...")
    chunk_mode = Config.CHUNK_MODE if Config.CHUNK_MODE in {"default", "encyclopedia"} else "default"
    print(f"Chunk mode: {chunk_mode}")

    source_paths = Config.SOURCE_HTML_PATHS or [Config.SOURCE_HTML_PATH]
    source_paths = [Path(p) for p in source_paths]
    print(f"Source HTML files: {[str(p) for p in source_paths]}")

    sourced_chunks = []
    for source_path in source_paths:
        source_chunks = parse_html_path_to_chunks(
            path=source_path,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            chunk_mode=chunk_mode,
        )
        sourced_chunks.extend((str(source_path), c) for c in source_chunks)
        print(f"  Parsed {len(source_chunks)} chunks from {source_path}")

    print(f"Total chunks to embed: {len(sourced_chunks)}")

    from app.embedder import embed

    chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)

    # Wipe and recreate collection for clean re-ingestion
    try:
        chroma_client.delete_collection(Config.CHROMA_COLLECTION)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=Config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    for i in range(0, len(sourced_chunks), batch_size):
        batch = sourced_chunks[i : i + batch_size]

        ids       = [f"{src}_{c.section_number}_{c.chunk_index}_{i+j}" for j, (src, c) in enumerate(batch)]
        documents = [c.document for _, c in batch]
        embeddings = [embed(c.document) for _, c in batch]
        metadatas = [
            {
                "source_path":   src,
                "section_number": c.section_number,
                "section_title":  c.section_title,
                "chunk_index":    c.chunk_index
            }
            for src, c in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"  Indexed chunks {i+1} to {min(i+batch_size, len(sourced_chunks))}")

    print(f"\nIngestion complete. Total chunks indexed: {collection.count()}")


if __name__ == "__main__":
    run_ingestion()
