"""llama-index-memory-geniffy through LlamaIndex's own FunctionAgent and Memory, with an LLM that answers from a
script and keeps what it was given, and stand-ins for the Geniffy clients: what the model sees, how often Geniffy
is asked, what is saved and where, the tools, and a memory that is away."""
from __future__ import annotations

import asyncio
import copy
from typing import Any, List

import pytest
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.mock import MockFunctionCallingLLM
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool

from llama_index.memory.geniffy import GeniffyMemoryBlock, geniffy_tools
from llama_index.memory.geniffy import base

CONTEXT = "- Priya Nair signs the Lumen renewal.  [Call with Priya, 1 Oct 2026]"


class Mem:
    """One user's memory, as client.space(...) gives it: records each call, with the space it was for."""

    def __init__(self, log: dict, space: Any, fail: bool = False):
        self.log, self.space_id, self.fail = log, space, fail
        self.memories = self

    def _check(self) -> None:
        if self.fail:
            raise ConnectionError("Geniffy is away")

    async def context(self, question: str) -> str:
        self._check()
        self.log["context"].append((self.space_id, question))
        return CONTEXT

    async def add(self, text: Any = None, *, messages: Any = None) -> dict:
        self._check()
        self.log["add"].append((self.space_id, text if messages is None else messages))
        return {"id": "src_1"}


class SyncMem(Mem):
    def context(self, question: str) -> str:  # type: ignore[override]
        self._check()
        self.log["context"].append((self.space_id, question))
        return CONTEXT

    def add(self, text: Any = None, *, messages: Any = None) -> dict:  # type: ignore[override]
        self._check()
        self.log["add"].append((self.space_id, text if messages is None else messages))
        return {"id": "src_1"}


class Client:
    def __init__(self, fail: bool = False, mem: type = Mem):
        self.log: dict = {"context": [], "add": []}
        self.fail, self.mem = fail, mem

    def space(self, space: str) -> Mem:
        return self.mem(self.log, space, self.fail)


def scripted(*replies: Any):
    """A function-calling LLM that answers from a script, one reply per call, and keeps what it was given."""
    queue, seen = list(replies), []

    def generate(messages: Any, **kwargs: Any) -> ChatMessage:
        seen.append(copy.deepcopy(list(messages)))
        reply = queue.pop(0)
        return reply if isinstance(reply, ChatMessage) else ChatMessage(role="assistant", content=reply)

    return MockFunctionCallingLLM(response_generator=generate), seen


def calls(tool: str, **kwargs: Any) -> ChatMessage:
    return ChatMessage(role="assistant", blocks=[ToolCallBlock(tool_call_id=f"call_{tool}", tool_name=tool,
                                                               tool_kwargs=kwargs)])


def system_of(messages: List[ChatMessage]) -> str:
    return "\n".join(m.content or "" for m in messages if str(m.role.value) == "system")


async def run(agent: FunctionAgent, block: GeniffyMemoryBlock, message: str, session: str = "s1") -> Any:
    memory = Memory.from_defaults(session_id=session, memory_blocks=[block])
    return await agent.run(user_msg=message, memory=memory)


def test_the_model_is_told_what_is_known_about_the_user():
    client = Client()
    llm, seen = scripted("Priya Nair signs it.")
    agent = FunctionAgent(llm=llm, system_prompt="You are a helpful assistant.")
    block = GeniffyMemoryBlock(space="user_1042", client=client)

    reply = asyncio.run(run(agent, block, "Who signs the Lumen renewal?"))

    assert str(reply) == "Priya Nair signs it."
    system = system_of(seen[0])
    assert "<geniffy>" in system and CONTEXT in system and "each line with where it came from" in system
    assert "You are a helpful assistant." in system, "your own system prompt stays"
    assert client.log["context"] == [("user_1042", "Who signs the Lumen renewal?")]
    assert client.log["add"] == [], "the block saves nothing by itself: remember() does"


def test_geniffy_is_asked_once_however_many_steps_a_run_takes():
    client = Client()
    lookups: List[str] = []

    async def lookup(q: str) -> str:
        """Look something up in the CRM."""
        lookups.append(q)
        return "Lumen: renewal due 1 Nov."

    llm, seen = scripted(calls("lookup", q="Lumen"), calls("lookup", q="Lumen owner"), "Priya Nair signs it.")
    agent = FunctionAgent(llm=llm, tools=[FunctionTool.from_defaults(async_fn=lookup)])
    asyncio.run(run(agent, GeniffyMemoryBlock(space="user_1042", client=client), "Who signs the Lumen renewal?"))

    assert lookups == ["Lumen", "Lumen owner"] and len(seen) == 3
    assert all(CONTEXT in system_of(step) for step in seen), "every step sees the memory"
    assert len(client.log["context"]) == 1, "and Geniffy is asked once"


def test_remember_saves_the_exchange_to_that_users_space():
    client = Client()
    llm, _ = scripted("Noted: you prefer WhatsApp.")
    agent = FunctionAgent(llm=llm)
    block = GeniffyMemoryBlock(space="user_1042", client=client)

    async def chat() -> None:
        reply = await run(agent, block, "Reach me on WhatsApp, not email.")
        await block.remember("Reach me on WhatsApp, not email.", reply)   # what agent.run returned, as is

    asyncio.run(chat())
    assert client.log["add"] == [("user_1042", [
        {"role": "user", "content": "Reach me on WhatsApp, not email."},
        {"role": "assistant", "content": "Noted: you prefer WhatsApp."},
    ])]

    asyncio.run(block.remember(ChatMessage(role="user", content="And call after 6."), "Will do."))
    assert client.log["add"][-1][1][0] == {"role": "user", "content": "And call after 6."}

    asyncio.run(block.remember("", "nothing was asked"))
    asyncio.run(block.remember("asked", ""))
    assert len(client.log["add"]) == 2, "an exchange with a side missing is not saved"


def test_a_memory_that_is_away_never_stops_the_agent():
    heard: List[Exception] = []
    llm, seen = scripted("I don't know who signs it.")
    agent = FunctionAgent(llm=llm, system_prompt="You are a helpful assistant.")
    block = GeniffyMemoryBlock(space="user_1042", client=Client(fail=True), on_error=heard.append)

    async def chat() -> Any:
        reply = await run(agent, block, "Who signs the Lumen renewal?")
        await block.remember("Who signs the Lumen renewal?", reply)
        return reply

    assert str(asyncio.run(chat())) == "I don't know who signs it."
    assert "<geniffy>" not in system_of(seen[0])
    assert [type(e) for e in heard] == [ConnectionError, ConnectionError], "the lookup and the save, both heard"


def test_without_on_error_a_memory_that_is_away_is_logged(caplog):
    block = GeniffyMemoryBlock(space="user_1042", client=Client(fail=True))
    with caplog.at_level("WARNING", logger="llama_index.memory.geniffy"):
        assert asyncio.run(block.aget([ChatMessage(role="user", content="hi")])) == ""
    assert "Geniffy is away" in caplog.text


def test_a_space_is_required_and_never_blank():
    with pytest.raises(TypeError, match="needs a space"):
        GeniffyMemoryBlock(client=Client())
    with pytest.raises(TypeError, match="needs a space"):
        geniffy_tools(client=Client())
    for blank in ("", "   "):
        with pytest.raises(ValueError, match="blank"):
            GeniffyMemoryBlock(space=blank, client=Client())
        with pytest.raises(ValueError, match="blank"):
            geniffy_tools(space=blank, client=Client())
    with pytest.raises(TypeError, match="string"):
        GeniffyMemoryBlock(space=1042, client=Client())
    with pytest.raises(TypeError, match="string"):
        geniffy_tools(space=1042, client=Client())


def test_space_none_is_your_own_memory_never_a_space():
    log: dict = {"context": [], "add": []}

    class Own:
        memories = None

        def space(self, space: str) -> Any:
            raise AssertionError("space=None must not pick a space")

        async def context(self, question: str) -> str:
            log["context"].append(question)
            return CONTEXT

    block = GeniffyMemoryBlock(space=None, client=Own())
    assert CONTEXT in asyncio.run(block.aget([ChatMessage(role="user", content="What did I decide?")]))
    assert log["context"] == ["What did I decide?"]


def test_the_overflow_from_short_term_memory_is_saved_only_when_asked():
    turns = [ChatMessage(role="user", content="I moved to Pune."),
             ChatMessage(role="assistant", blocks=[ToolCallBlock(tool_call_id="c", tool_name="t", tool_kwargs={})]),
             ChatMessage(role="tool", content="ok", additional_kwargs={"tool_call_id": "c"}),
             ChatMessage(role="assistant", content="Noted.")]

    client = Client()
    asyncio.run(GeniffyMemoryBlock(space="user_1042", client=client).aput(turns, from_short_term_memory=True))
    assert client.log["add"] == [], "by default each exchange is saved with remember(), not twice"

    asyncio.run(GeniffyMemoryBlock(space="user_1042", client=client, accept_short_term_memory=True)
                .aput(turns, from_short_term_memory=True))
    assert client.log["add"] == [("user_1042", [{"role": "user", "content": "I moved to Pune."},
                                                {"role": "assistant", "content": "Noted."}])], \
        "tool calls and their results are left out"


def test_a_question_asked_again_after_a_save_is_looked_up_afresh():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client)
    asked = [ChatMessage(role="user", content="Where do I live?")]

    async def go() -> None:
        await block.aget(asked)
        await block.aget(asked)
        assert len(client.log["context"]) == 1
        await block.remember("I moved to Pune.", "Noted.")
        await block.aget(asked)

    asyncio.run(go())
    assert len(client.log["context"]) == 2


def test_nothing_is_looked_up_without_a_user_message():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client)
    assert asyncio.run(block.aget([])) == ""
    assert asyncio.run(block.aget([ChatMessage(role="assistant", content="Hello!")])) == ""
    assert client.log["context"] == []


def test_instructions_can_be_changed_or_left_out():
    block = GeniffyMemoryBlock(space="user_1042", client=Client(), instructions="Facts about the user:")
    assert asyncio.run(block.aget([ChatMessage(role="user", content="hi")])) == f"Facts about the user:\n\n{CONTEXT}"
    bare = GeniffyMemoryBlock(space="user_1042", client=Client(), instructions="")
    assert asyncio.run(bare.aget([ChatMessage(role="user", content="hi")])) == CONTEXT


def test_the_tools_read_and_save_for_one_user():
    client = Client()
    recall, remember = geniffy_tools(space="user_1042", client=client)
    assert (recall.metadata.name, remember.metadata.name) == ("recall", "remember")
    assert "When nothing is known, it says so" in recall.metadata.description
    assert "Never passwords" in remember.metadata.description

    llm, seen = scripted(calls("recall", query="Lumen renewal"), "Priya Nair signs it.")
    agent = FunctionAgent(llm=llm, tools=[recall, remember])

    async def ask() -> Any:
        return await agent.run(user_msg="Who signs the Lumen renewal?")

    reply = asyncio.run(ask())

    assert str(reply) == "Priya Nair signs it."
    assert any(CONTEXT in (m.content or "") for m in seen[1] if str(m.role.value) == "tool"), \
        "the model gets what recall found"
    assert client.log["context"] == [("user_1042", "Lumen renewal")]

    assert str(remember.call(text="Prefers WhatsApp.")) == "Saved."
    assert client.log["add"] == [("user_1042", "Prefers WhatsApp.")]


def test_a_sync_client_works_too():
    client = Client(mem=SyncMem)
    block = GeniffyMemoryBlock(space="user_1042", client=client)
    assert CONTEXT in asyncio.run(block.aget([ChatMessage(role="user", content="hi")]))
    asyncio.run(block.remember("hi", "hello"))
    recall, _ = geniffy_tools(space="user_1042", client=client)
    assert str(recall.call(query="q")) == CONTEXT
    assert len(client.log["context"]) == 2 and len(client.log["add"]) == 1


def test_without_a_client_one_is_made_for_each_event_loop(monkeypatch):
    import geniffy

    made: List[Any] = []

    class Fake(Client):
        def __init__(self, *a: Any, **kw: Any):
            super().__init__()
            made.append(self)

    monkeypatch.setattr(geniffy, "AsyncGeniffy", Fake)
    monkeypatch.setattr(base, "_shared", {})

    async def twice() -> None:
        for space in ("user_1", "user_2"):
            block = GeniffyMemoryBlock(space=space)
            await block.aget([ChatMessage(role="user", content="hi")])

    asyncio.run(twice())
    assert len(made) == 1, "one client for the loop, shared by every block"
    assert made[0].log["context"] == [("user_1", "hi"), ("user_2", "hi")]

    asyncio.run(twice())
    assert len(made) == 2, "a new loop gets its own"
    assert len(base._shared) == 1, "and the closed loop's client is let go"
