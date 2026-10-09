"""GeniffyMemoryBlock with the real geniffy client, its calls answered by a stand-in for the API: what goes over the
wire for the briefing, for a run saved into its conversation, and for an account without the briefing switched on."""
from __future__ import annotations

import asyncio
import json
from typing import Any, List

import httpx
from geniffy import AsyncGeniffy
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory

from llama_index.memory.geniffy import GeniffyMemoryBlock, __version__
from test_memory_geniffy import BRIEFING, CONTEXT, scripted, system_of

SOURCE = {"id": "a" * 32, "kind": "note", "title": "Who signs the Lumen renewal?", "status": "reading"}


class Api:
    """The API, as far as the block reaches it: the briefing (or not switched on), context and adding."""

    def __init__(self, briefing_on: bool = True):
        self.briefing_on = briefing_on
        self.calls: List[Any] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.url.path, request.headers.get("x-geniffy-space"), body))
        assert f"llama-index-memory-geniffy/{__version__}" in request.headers["user-agent"]
        if request.url.path == "/v1/briefing":
            if not self.briefing_on:
                return httpx.Response(404, json={"error": {"code": "not_switched_on", "message": (
                    "The briefing, where things stand, episodes, lessons and intentions aren't switched on for "
                    "this memory yet.")}})
            return httpx.Response(200, json={"project": body.get("project"), "briefing": BRIEFING, "now": [],
                                             "due": [], "lessons": [], "episodes": [], "memories": []})
        if request.url.path == "/v1/context":
            return httpx.Response(200, json={"context": CONTEXT, "memories": [], "found": True})
        if request.url.path == "/v1/memories":
            return httpx.Response(201, json={"source": SOURCE})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "No such door."}})


def chat(api: Api, *replies: str) -> List[Any]:
    llm, seen = scripted(*replies)
    agent = FunctionAgent(llm=llm)

    async def go() -> None:
        client = AsyncGeniffy("gnf_test", base_url="https://api.test",
                              integration=f"llama-index-memory-geniffy/{__version__}",
                              http_client=httpx.AsyncClient(transport=httpx.MockTransport(api)), max_retries=0)
        block = GeniffyMemoryBlock(space="user_1042", project="lumen", client=client)
        memory = Memory.from_defaults(session_id="chat-7", memory_blocks=[block])
        await agent.run(user_msg="Who signs the Lumen renewal?", memory=memory)
        await block.save(memory)

    asyncio.run(go())
    return seen


def test_the_briefing_and_the_conversation_as_they_go_over_the_wire():
    api = Api()
    chat(api, "Priya Nair signs it.")
    assert api.calls == [
        ("/v1/briefing", "user_1042",
         {"project": "lumen", "cue": "Who signs the Lumen renewal?", "budget_chars": 6000}),
        ("/v1/memories", "user_1042", {"messages": [{"role": "user", "content": "Who signs the Lumen renewal?"},
                                                    {"role": "assistant", "content": "Priya Nair signs it."}],
                                       "session": "chat-7", "labels": {"project": "lumen"},
                                       "title": "Who signs the Lumen renewal?"}),
    ]


def test_an_account_without_the_briefing_reads_context_over_the_wire():
    api = Api(briefing_on=False)
    seen = chat(api, "Priya Nair signs it.")
    assert [path for path, _, _ in api.calls] == ["/v1/briefing", "/v1/context", "/v1/memories"]
    assert CONTEXT in system_of(seen[0])
