"""Ask a question against the local RAG index."""

import argparse
import asyncio

from rag import get_rag_response


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question using the local RAG index.")
    parser.add_argument("query", help="Question to answer from the indexed documents")
    args = parser.parse_args()
    result = await get_rag_response(args.query)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
