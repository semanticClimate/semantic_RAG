import chromadb
from pathlib import Path
from config import Config
from html_sectioning import parse_html_path_to_chunks

def _ingest_single_source(chroma_client, html_path: Path, collection_name: str, source_name: str):
    print(f"\nIngesting source '{source_name}' from: {html_path}")
    chunks = parse_html_path_to_chunks(
        path=html_path,
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    print(f"Total chunks to embed ({source_name}): {len(chunks)}")
    from app.embedder import embed

    # Wipe and recreate collection for clean re-ingestion
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        ids       = [f"{source_name}_{c.section_number}_{c.chunk_index}_{i+j}" for j, c in enumerate(batch)]
        documents = [c.document for c in batch]
        embeddings = [embed(c.document) for c in batch]
        metadatas = [
            {
                "section_number": c.section_number,
                "section_title":  c.section_title,
                "chunk_index":    c.chunk_index,
                "source":         source_name
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"  Indexed chunks {i+1} to {min(i+batch_size, len(chunks))}")

    print(f"Ingestion complete for '{source_name}'. Total chunks indexed: {collection.count()}")


def run_ingestion():
    print("Starting ingestion...")
    chroma_client = chromadb.PersistentClient(path=Config.CHROMA_PATH)

    # Existing source (book) collection
    _ingest_single_source(
        chroma_client=chroma_client,
        html_path=Path(Config.CLIMATE_BOOK_HTML_PATH),
        collection_name=Config.CHROMA_COLLECTION,
        source_name="climate_book"
    )

    # New source (encyclopedia) collection
    encyclopedia_path = Path(Config.CLIMATE_ENCYCLOPEDIA_HTML_PATH)
    if encyclopedia_path.exists():
        _ingest_single_source(
            chroma_client=chroma_client,
            html_path=encyclopedia_path,
            collection_name=Config.CHROMA_COLLECTION_CLIMATE_HTML,
            source_name="climate_encyclopedia"
        )
    else:
        print(
            f"Skipped encyclopedia ingestion: file not found at "
            f"{encyclopedia_path.resolve()}"
        )


if __name__ == "__main__":
    run_ingestion()
