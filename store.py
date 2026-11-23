"""One embedding configuration for both indexing and retrieval."""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

COLLECTION = "technical_notes"
EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingModelMismatch(ValueError):
    """The persisted index was built with a different embedding model."""


def open_store(
    db_path: str | Path,
    *,
    embeddings: Embeddings | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> Chroma:
    load_dotenv()
    if embeddings is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY in .env before indexing or querying.")
        embeddings = OpenAIEmbeddings(model=embedding_model)

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(
        COLLECTION, metadata={"embedding_model": embedding_model}
    )
    existing_model = (collection.metadata or {}).get("embedding_model")
    if existing_model != embedding_model:
        raise EmbeddingModelMismatch(
            f"Index uses {existing_model!r}, but this process requested {embedding_model!r}. "
            "Rebuild the index in a new directory or use the original model."
        )
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
