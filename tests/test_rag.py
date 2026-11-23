import json
import unittest

from langchain_core.runnables import RunnableLambda

from rag import build_chain
from schemas import RagResponse
from tests.test_ingest import LocalEmbeddings, temporary_workspace
from ingest import ingest_documents
from store import open_store


class RagTests(unittest.IsolatedAsyncioTestCase):
    async def test_answerable_question_has_source_reference(self):
        async def model(messages):
            prompt = str(messages)
            self.assertIn("Redis caches summaries for 60 seconds", prompt)
            return json.dumps(
                {"texto": "Redis caches order summaries for 60 seconds.", "referencias": ["cache.md"]}
            )

        with temporary_workspace() as root:
            data = root / "data"
            data.mkdir()
            (data / "cache.md").write_text("Redis caches summaries for 60 seconds.", encoding="utf-8")
            db = root / "db"
            embeddings = LocalEmbeddings()
            ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            store = open_store(db, embeddings=embeddings, embedding_model="test-v1")
            result = await build_chain(store, RunnableLambda(model)).ainvoke("How long does Redis cache summaries?")
            self.assertIsInstance(result, RagResponse)
            self.assertIn("60 seconds", result.texto)
            self.assertEqual(result.referencias, ["cache.md"])
            del store

    async def test_trap_question_abstains(self):
        async def model(messages):
            prompt = str(messages)
            self.assertIn("No lo sé", prompt)
            return json.dumps({"texto": "No lo sé", "referencias": []})

        with temporary_workspace() as root:
            data = root / "data"
            data.mkdir()
            (data / "cache.md").write_text("Redis caches summaries for 60 seconds.", encoding="utf-8")
            db = root / "db"
            embeddings = LocalEmbeddings()
            ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            store = open_store(db, embeddings=embeddings, embedding_model="test-v1")
            result = await build_chain(store, RunnableLambda(model)).ainvoke("What city hosts the server?")
            self.assertEqual(result.texto, "No lo sé")
            self.assertEqual(result.referencias, [])
            del store

    async def test_rejects_unretrieved_reference(self):
        async def model(_messages):
            return json.dumps({"texto": "An unsupported claim", "referencias": ["secret.md"]})

        with temporary_workspace() as root:
            data = root / "data"
            data.mkdir()
            (data / "cache.md").write_text("Redis caches summaries for 60 seconds.", encoding="utf-8")
            db = root / "db"
            embeddings = LocalEmbeddings()
            ingest_documents(data, db, embeddings=embeddings, embedding_model="test-v1")
            store = open_store(db, embeddings=embeddings, embedding_model="test-v1")
            with self.assertRaises(ValueError):
                await build_chain(store, RunnableLambda(model)).ainvoke("How long is the cache TTL?")
            del store
