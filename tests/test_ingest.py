import gc
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from chromadb.api.client import SharedSystemClient
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

from ingest import CHUNK_TOKENS, OVERLAP_TOKENS, ingest_documents
from store import EmbeddingModelMismatch, open_store


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        return [float(words.count(word)) for word in ("fastapi", "redis", "postgresql", "health")]


@contextmanager
def temporary_workspace():
    with tempfile.TemporaryDirectory() as temp:
        try:
            yield Path(temp)
        finally:
            SharedSystemClient.clear_system_cache()
            gc.collect()


class IngestionTests(unittest.TestCase):
    def test_splitter_uses_500_tokens_with_50_token_overlap(self):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=CHUNK_TOKENS,
            chunk_overlap=OVERLAP_TOKENS,
        )
        chunks = splitter.split_text(" ".join(f"event{i}" for i in range(650)))
        encoder = tiktoken.get_encoding("cl100k_base")
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(encoder.encode(chunks[0])), 500)
        self.assertEqual(len(encoder.encode(chunks[1])), 500)
        self.assertEqual(
            encoder.encode(" ".join(chunks[0].split()[-25:])),
            encoder.encode(" ".join(chunks[1].split()[:25])),
        )

    def test_ingestion_persists_and_skips_unchanged_chunks(self):
        with temporary_workspace() as root:
            data = root / "data"
            data.mkdir()
            (data / "notes.md").write_text("FastAPI reads PostgreSQL. Redis caches summaries.", encoding="utf-8")
            db = root / "db"
            embeddings = LocalEmbeddings()

            first = ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            second = ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            reopened = open_store(db, embeddings=embeddings, embedding_model="test-v1")

            self.assertEqual(first.added, 1)
            self.assertEqual(second.added, 0)
            self.assertEqual(second.removed, 0)
            self.assertEqual(len(reopened.get()["ids"]), 1)
            self.assertEqual(reopened.similarity_search("Redis", k=1)[0].metadata["source"], "notes.md")
            del reopened

    def test_changed_file_replaces_old_chunks_and_embedding_mismatch_fails(self):
        with temporary_workspace() as root:
            data = root / "data"
            data.mkdir()
            file = data / "notes.txt"
            file.write_text("Redis caches summaries.", encoding="utf-8")
            db = root / "db"
            embeddings = LocalEmbeddings()
            ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            file.write_text("PostgreSQL stores orders.", encoding="utf-8")

            result = ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            self.assertEqual((result.added, result.removed), (1, 1))
            with self.assertRaises(EmbeddingModelMismatch):
                open_store(db, embeddings=embeddings, embedding_model="different")

    def test_missing_data_does_not_create_database(self):
        with temporary_workspace() as root:
            with self.assertRaises(ValueError):
                ingest_documents(root / "missing", root / "db", embeddings=LocalEmbeddings(), embedding_model="test-v1")
            self.assertFalse((root / "db").exists())
