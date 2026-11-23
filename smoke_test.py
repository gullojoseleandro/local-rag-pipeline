"""Optional live smoke test; requires an OpenAI API key and incurs API usage."""

import asyncio

from rag import get_rag_response


async def main() -> None:
    answer = await get_rag_response("How long are order summaries cached in Redis?")
    print("Answerable:", answer.model_dump_json())
    if "60 seconds" not in answer.texto or "cache.md" not in answer.referencias:
        raise AssertionError("Expected the documented Redis TTL and source reference.")

    trap = await get_rag_response("Which city hosts the API server?")
    print("Trap:", trap.model_dump_json())
    if trap.texto.strip() != "No lo sé" or trap.referencias:
        raise AssertionError("Expected an abstention without references.")


if __name__ == "__main__":
    asyncio.run(main())
