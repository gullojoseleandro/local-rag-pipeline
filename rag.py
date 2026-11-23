"""Asynchronous LCEL retrieval-augmented generation."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from schemas import RagResponse
from store import open_store

TOP_K = 4


def build_chain(store, model: Runnable):
    """Compose retrieval, contextual prompt, async model call, and validation."""
    parser = PydanticOutputParser(pydantic_object=RagResponse)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a technical assistant. Answer using ONLY the provided context. "
                "If the context does not contain the answer, respond with exactly "
                "'No lo sé' and an empty referencias list. Do not use outside knowledge. "
                "Cite only source filenames that support your answer. "
                "Return only a JSON object matching these instructions:\n{format_instructions}\n\n"
                "CONTEXT:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    def prepare(payload):
        documents = payload["documents"]
        sources = {document.metadata["source"] for document in documents}
        context = "\n\n".join(
            f"[source: {document.metadata['source']}]\n{document.page_content}"
            for document in documents
        )
        return {
            "question": payload["question"],
            "context": context,
            "sources": sources,
            "format_instructions": parser.get_format_instructions(),
        }

    def validate(payload):
        result: RagResponse = payload["result"]
        if not set(result.referencias).issubset(payload["sources"]):
            raise ValueError("The answer cites a document that was not retrieved.")
        return result

    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    return (
        RunnableParallel(question=RunnablePassthrough(), documents=retriever)
        | RunnableLambda(prepare)
        | RunnablePassthrough.assign(result=prompt | model | parser)
        | RunnableLambda(validate)
    )


async def get_rag_response(query: str) -> RagResponse:
    """Retrieve from the persisted local index and answer asynchronously."""
    if not query.strip():
        raise ValueError("Query must not be empty.")
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in .env before querying.")
    db_path = Path(os.getenv("RAG_DB_PATH", "vectorstore"))
    if not db_path.is_dir():
        raise ValueError("Index missing. Run 'python ingest.py' first.")
    store = open_store(db_path)
    if not store.get(include=[])["ids"]:
        raise ValueError("Index is empty. Run 'python ingest.py' first.")
    model = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    return await build_chain(store, model).ainvoke(query)
