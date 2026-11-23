"""Load Markdown/text documents into an idempotent local Chroma index."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from store import EMBEDDING_MODEL, open_store

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 50


@dataclass(frozen=True)
class IngestResult:
    added: int
    removed: int
    total: int


def _read_documents(data_dir: Path) -> list[Document]:
    if not data_dir.is_dir():
        raise ValueError(f"Data directory not found: {data_dir}")
    files = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    if not files:
        raise ValueError(f"No .md or .txt files found in {data_dir}")
    documents = [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(data_dir).as_posix()},
        )
        for path in files
    ]
    return [document for document in documents if document.page_content.strip()]


def ingest_documents(
    data_dir: str | Path = "data",
    db_path: str | Path = "vectorstore",
    *,
    embeddings: Embeddings | None = None,
    embedding_model: str = EMBEDDING_MODEL,
) -> IngestResult:
    documents = _read_documents(Path(data_dir))
    if not documents:
        raise ValueError("All input documents are empty.")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_TOKENS,
        chunk_overlap=OVERLAP_TOKENS,
    )
    chunks = splitter.split_documents(documents)
    desired: dict[str, Document] = {}
    for chunk in chunks:
        digest = hashlib.sha256(
            f"{chunk.metadata['source']}\0{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        desired[digest] = chunk

    store = open_store(db_path, embeddings=embeddings, embedding_model=embedding_model)
    existing = set(store.get(include=[])["ids"])
    additions = sorted(set(desired) - existing)
    removals = sorted(existing - set(desired))
    if additions:
        store.add_documents([desired[item] for item in additions], ids=additions)
    if removals:
        store.delete(ids=removals)
    return IngestResult(len(additions), len(removals), len(desired))


def main() -> None:
    parser = argparse.ArgumentParser(description="Index local Markdown and text files.")
    parser.add_argument("--data", default="data", help="Source document directory")
    parser.add_argument("--db", default="vectorstore", help="Persistent Chroma directory")
    args = parser.parse_args()
    result = ingest_documents(args.data, args.db)
    print(f"Indexed: {result.total} chunks ({result.added} added, {result.removed} removed).")


if __name__ == "__main__":
    main()
