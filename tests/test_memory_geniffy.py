"""llama-index-memory-geniffy through LlamaIndex's own FunctionAgent and Memory, with an LLM that answers from a
script and keeps what it was given, and stand-ins for the Geniffy clients: what the model sees, how often Geniffy
is asked, what is saved and where (tool calls included, one memory per conversation, never twice), an account
without the briefing, the tools, and a memory that is away."""
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
BRIEFING = ("Where things stand (lumen), as of 2 Oct:\n- Next: send Priya the renewal\n"
            "What is known (facts):\n- [2026-10-01] Priya Nair signs the Lumen renewal.")


class Status(Exception):
    """An error from the API, with its status, as the geniffy SDK raises them."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Mem:
    """One user's memory, as client.space(...) gives it: records each call, with the space it was for."""

    def __init__(self, client: "Client", space: Any):
        self.client, self.log, self.space_id = client, client.log, space
        self.memories = self

    def _check(self) -> None:
        if self.client.fail:
            raise ConnectionError("Geniffy is away")

    def _briefing(self, project: Any, cue: str) -> str:
        self._check()
        self.log["briefing"].append((self.space_id, project, cue))
        if isinstance(self.client.briefing, Exception):
            raise self.client.briefing
        return self.client.briefing

    def _context(self, question: str) -> str:
        self._check()
        self.log["context"].append((self.space_id, question))
        return CONTEXT

    def _add(self, text: Any, messages: Any, session: Any, title: Any, labels: Any) -> dict:
        self._check()
        if messages is None:
            self.log["add"].append((self.space_id, text))
            return {"id": "src_1"}
        if self.client.saves_fail:
            self.client.saves_fail -= 1
            raise Status(502, "Your memory didn't answer. Try again in a minute.")
        self.log["session"].append({"space": self.space_id, "session": session, "title": title, "labels": labels,
                                    "messages": messages})
        return {"id": "src_1"}

    async def briefing(self, *, project: Any = None, cue: str = "", budget_chars: int = 6000) -> str:
        return self._briefing(project, cue)

    async def context(self, question: str) -> str:
        return self._context(question)

    async def add(self, text: Any = None, *, messages: Any = None, session: Any = None, title: Any = None,
                  labels: Any = None) -> dict:
        return self._add(text, messages, session, title, labels)


class SyncMem(Mem):
    def briefing(self, *, project: Any = None, cue: str = "",  # type: ignore[override]
                 budget_chars: int = 6000) -> str:
        return self._briefing(project, cue)

    def context(self, question: str) -> str:  # type: ignore[override]
        return self._context(question)

    def add(self, text: Any = None, *, messages: Any = None,  # type: ignore[override]
            session: Any = None, title: Any = None, labels: Any = None) -> dict:
        return self._add(text, messages, session, title, labels)


class Client:
    """A stand-in for geniffy.AsyncGeniffy. briefing is what the briefing says, or the error it raises; saves_fail is
    how many of the next saves fail."""

    def __init__(self, fail: bool = False, mem: type = Mem, briefing: Any = BRIEFING, saves_fail: int = 0):
        self.log: dict = {"briefing": [], "context": [], "add": [], "session": []}
        self.fail, self.mem, self.briefing, self.saves_fail = fail, mem, briefing, saves_fail

    def space(self, space: str) -> Mem:
        return self.mem(self, space)


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


async def crm(q: str) -> str:
    """Look something up in the CRM."""
    return "Lumen: renewal due 1 Nov."


def test_the_model_is_given_the_users_briefing():
    client = Client()
    llm, seen = scripted("Priya Nair signs it.")
    agent = FunctionAgent(llm=llm, system_prompt="You are a helpful assistant.")
    block = GeniffyMemoryBlock(space="user_1042", client=client)

    reply = asyncio.run(run(agent, block, "Who signs the Lumen renewal?"))

    assert str(reply) == "Priya Nair signs it."
    system = system_of(seen[0])
    assert "<geniffy>" in system and BRIEFING in system and "each line with its date" in system
    assert "You are a helpful assistant." in system, "your own system prompt stays"
    assert client.log["briefing"] == [("user_1042", None, "Who signs the Lumen renewal?")]
    assert client.log["context"] == [], "the briefing holds what is known that bears on the question"
    assert client.log["session"] == [], "the block saves nothing by itself: save() does"


def test_geniffy_is_asked_once_however_many_steps_a_run_takes():
    client = Client()
    llm, seen = scripted(calls("crm", q="Lumen"), calls("crm", q="Lumen owner"), "Priya Nair signs it.")
    agent = FunctionAgent(llm=llm, tools=[FunctionTool.from_defaults(async_fn=crm)])
    asyncio.run(run(agent, GeniffyMemoryBlock(space="user_1042", client=client), "Who signs the Lumen renewal?"))

    assert len(seen) == 3
    assert all(BRIEFING in system_of(step) for step in seen), "every step sees the memory"
    assert len(client.log["briefing"]) == 1, "and Geniffy is asked once"


def test_save_keeps_the_whole_run_with_its_tool_calls_in_one_memory_for_the_conversation():
    client = Client()
    llm, _ = scripted(calls("crm", q="Lumen"), "Priya Nair signs it, due 1 Nov.", "Send it Monday.")
    agent = FunctionAgent(llm=llm, tools=[FunctionTool.from_defaults(async_fn=crm)])
    block = GeniffyMemoryBlock(space="user_1042", project="lumen", client=client)

    async def chat() -> None:
        memory = Memory.from_defaults(session_id="chat-7", memory_blocks=[block])
        await agent.run(user_msg="Who signs the Lumen renewal?", memory=memory)
        await block.save(memory)
        await block.save(memory)            # saved twice by mistake: nothing new is sent
        await agent.run(user_msg="When should I send it?", memory=memory)
        await block.save(memory)

    asyncio.run(chat())
    first, second = client.log["session"]
    assert (first["space"], first["session"], first["labels"]) == ("user_1042", "chat-7", {"project": "lumen"})
    assert first["title"] == "Who signs the Lumen renewal?"
    assert first["messages"] == [
        {"role": "user", "content": "Who signs the Lumen renewal?"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "crm", "args": {"q": "Lumen"},
                                                             "id": "call_crm"}]},
        {"role": "tool", "content": "Lumen: renewal due 1 Nov.", "tool_call_id": "call_crm"},
        {"role": "assistant", "content": "Priya Nair signs it, due 1 Nov."},
    ], "what the agent did is how Geniffy learns what happened"
    assert second["session"] == "chat-7" and second["messages"] == [
        {"role": "user", "content": "When should I send it?"}, {"role": "assistant", "content": "Send it Monday."}]
    assert [b[1] for b in client.log["briefing"]] == ["lumen", "lumen"]


def test_a_save_that_fails_goes_with_the_next_save():
    heard: List[Exception] = []
    client = Client(saves_fail=1)
    llm, _ = scripted("Priya Nair signs it.", "In March.")
    agent = FunctionAgent(llm=llm)
    block = GeniffyMemoryBlock(space="user_1042", client=client, on_error=heard.append)

    async def chat() -> None:
        memory = Memory.from_defaults(session_id="chat-8", memory_blocks=[block])
        for message in ("Who signs the Lumen renewal?", "And when is it due?"):
            await agent.run(user_msg=message, memory=memory)
            await block.save(memory)

    asyncio.run(chat())
    assert [e.status for e in heard] == [502]
    [saved] = client.log["session"]
    assert [m["content"] for m in saved["messages"]] == [
        "Who signs the Lumen renewal?", "Priya Nair signs it.", "And when is it due?", "In March."]


def test_save_takes_the_messages_your_app_keeps_and_its_own_id():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client)
    history = [ChatMessage(role="user", content="Hi"), ChatMessage(role="assistant", content="Hello!"),
               ChatMessage(role="user", content="Who signs it?"), ChatMessage(role="assistant", content="Priya.")]
    asyncio.run(block.save(history, session="chat-9"))
    assert client.log["session"][0]["session"] == "chat-9"
    assert client.log["session"][0]["messages"] == [{"role": "user", "content": "Who signs it?"},
                                                    {"role": "assistant", "content": "Priya."}], "this turn only"


def test_remember_saves_one_exchange_to_that_users_space():
    client = Client()
    llm, _ = scripted("Noted: you prefer WhatsApp.")
    agent = FunctionAgent(llm=llm)
    block = GeniffyMemoryBlock(space="user_1042", client=client)

    async def chat() -> None:
        reply = await run(agent, block, "Reach me on WhatsApp, not email.")
        await block.remember("Reach me on WhatsApp, not email.", reply)   # what agent.run returned, as is

    asyncio.run(chat())
    [saved] = client.log["session"]
    assert saved["space"] == "user_1042" and saved["session"].startswith("run-"), "a memory of its own"
    assert saved["messages"] == [{"role": "user", "content": "Reach me on WhatsApp, not email."},
                                 {"role": "assistant", "content": "Noted: you prefer WhatsApp."}]

    asyncio.run(block.remember(ChatMessage(role="user", content="And call after 6."), "Will do.", session="chat-7"))
    assert client.log["session"][-1]["session"] == "chat-7"
    assert client.log["session"][-1]["messages"][0] == {"role": "user", "content": "And call after 6."}

    asyncio.run(block.remember("", "nothing was asked"))
    asyncio.run(block.remember("asked", ""))
    assert len(client.log["session"]) == 2, "an exchange with a side missing is not saved"


def test_a_memory_that_is_away_never_stops_the_agent():
    heard: List[Exception] = []
    llm, seen = scripted("I don't know who signs it.")
    agent = FunctionAgent(llm=llm, system_prompt="You are a helpful assistant.")
    block = GeniffyMemoryBlock(space="user_1042", client=Client(fail=True), on_error=heard.append)

    async def chat() -> Any:
        memory = Memory.from_defaults(session_id="s1", memory_blocks=[block])
        reply = await agent.run(user_msg="Who signs the Lumen renewal?", memory=memory)
        await block.save(memory)
        return reply

    assert str(asyncio.run(chat())) == "I don't know who signs it."
    assert "<geniffy>" not in system_of(seen[0])
    assert [type(e) for e in heard] == [ConnectionError, ConnectionError], "the lookup and the save, both heard"


def test_without_on_error_a_memory_that_is_away_is_logged(caplog):
    block = GeniffyMemoryBlock(space="user_1042", client=Client(fail=True))
    with caplog.at_level("WARNING", logger="llama_index.memory.geniffy"):
        assert asyncio.run(block.aget([ChatMessage(role="user", content="hi")])) == ""
    assert "Geniffy is away" in caplog.text


def test_without_the_briefing_switched_on_it_reads_what_is_known_and_tries_again_later():
    client = Client(briefing=Status(404, "The briefing isn't switched on for this memory yet."))
    block = GeniffyMemoryBlock(space="user_1042", client=client, on_error=lambda e: pytest.fail(f"not an error: {e}"))

    async def ask(question: str) -> str:
        return await block.aget([ChatMessage(role="user", content=question)])

    assert CONTEXT in asyncio.run(ask("Who signs the Lumen renewal?"))
    asyncio.run(ask("Where do I live?"))
    assert len(client.log["briefing"]) == 1, "asked once, then not for ten minutes"
    assert [q for _, q in client.log["context"]] == ["Who signs the Lumen renewal?", "Where do I live?"]

    client.briefing, block._off_until = BRIEFING, 0.0       # ten minutes on, and it is switched on
    assert BRIEFING in asyncio.run(ask("And when is it due?"))


def test_a_briefing_that_fails_another_way_is_an_error_not_a_reason_to_read_something_else():
    heard: List[Exception] = []
    client = Client(briefing=Status(502, "Your memory didn't answer."))
    block = GeniffyMemoryBlock(space="user_1042", client=client, on_error=heard.append)
    assert asyncio.run(block.aget([ChatMessage(role="user", content="hi")])) == ""
    assert [e.status for e in heard] == [502] and client.log["context"] == []


def test_an_empty_briefing_still_says_nothing_is_remembered_yet():
    block = GeniffyMemoryBlock(space="user_1042", client=Client(briefing=""), instructions="")
    assert asyncio.run(block.aget([ChatMessage(role="user", content="What is my name?")])) == \
        "Nothing is remembered about this user yet."


def test_briefing_false_reads_only_what_bears_on_the_question():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client, briefing=False, instructions="")
    assert asyncio.run(block.aget([ChatMessage(role="user", content="Who signs it?")])) == CONTEXT
    assert asyncio.run(block.aget([])) == "", "and nothing without a question"
    assert client.log["briefing"] == [] and client.log["context"] == [("user_1042", "Who signs it?")]


def test_without_a_user_message_the_briefing_still_says_where_things_stand():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client, instructions="")
    assert asyncio.run(block.aget([])) == BRIEFING
    assert client.log["briefing"] == [("user_1042", None, "")] and client.log["context"] == []


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
    log: list = []

    class Own:
        memories = None

        def space(self, space: str) -> Any:
            raise AssertionError("space=None must not pick a space")

        async def briefing(self, **kwargs: Any) -> str:
            log.append(kwargs["cue"])
            return BRIEFING

    block = GeniffyMemoryBlock(space=None, client=Own())
    assert BRIEFING in asyncio.run(block.aget([ChatMessage(role="user", content="What did I decide?")]))
    assert log == ["What did I decide?"]


def test_the_overflow_from_short_term_memory_is_saved_only_when_asked_into_its_conversation():
    turns = [ChatMessage(role="user", content="I moved to Pune."),
             ChatMessage(role="assistant", blocks=[ToolCallBlock(tool_call_id="c", tool_name="t", tool_kwargs={})]),
             ChatMessage(role="tool", content="ok", additional_kwargs={"tool_call_id": "c"}),
             ChatMessage(role="assistant", content="Noted.")]

    client = Client()
    asyncio.run(GeniffyMemoryBlock(space="user_1042", client=client)
                .aput(copy.deepcopy(turns), from_short_term_memory=True, session_id="chat-3"))
    assert client.log["session"] == [], "by default each run is saved with save(), not twice"

    asyncio.run(GeniffyMemoryBlock(space="user_1042", client=client, accept_short_term_memory=True)
                .aput(copy.deepcopy(turns), from_short_term_memory=True, session_id="chat-3"))
    [saved] = client.log["session"]
    assert saved["session"] == "chat-3"
    assert saved["messages"] == [{"role": "user", "content": "I moved to Pune."},
                                 {"role": "assistant", "content": "", "tool_calls": [{"name": "t", "args": {},
                                                                                      "id": "c"}]},
                                 {"role": "tool", "content": "ok", "tool_call_id": "c"},
                                 {"role": "assistant", "content": "Noted."}]


def test_messages_are_sent_as_geniffy_reads_them_and_no_larger_than_it_keeps():
    openai_style = ChatMessage(role="assistant", content="Looking.", additional_kwargs={"tool_calls": [
        {"id": "c9", "type": "function", "function": {"name": "fetch", "arguments": '{"url": "https://x.test"}'}}]})
    assert base._as_sent(openai_style) == {"role": "assistant", "content": "Looking.", "tool_calls": [
        {"name": "fetch", "args": '{"url": "https://x.test"}', "id": "c9"}]}
    page = base._as_sent(ChatMessage(role="tool", content="x" * 10_000,
                                     additional_kwargs={"tool_call_id": "c9", "name": "fetch"}))
    assert len(page["content"]) == 4_000 and page["name"] == "fetch"
    assert base._as_sent(ChatMessage(role="system", content="You are a helpful assistant.")) is None
    assert base._as_sent(ChatMessage(role="model", content="Hi.")) == {"role": "assistant", "content": "Hi."}


def test_a_question_asked_again_after_a_save_is_looked_up_afresh():
    client = Client()
    block = GeniffyMemoryBlock(space="user_1042", client=client)
    asked = [ChatMessage(role="user", content="Where do I live?")]

    async def go() -> None:
        await block.aget(asked)
        await block.aget(asked)
        assert len(client.log["briefing"]) == 1
        await block.remember("I moved to Pune.", "Noted.")
        await block.aget(asked)

    asyncio.run(go())
    assert len(client.log["briefing"]) == 2


def test_instructions_can_be_changed_or_left_out():
    block = GeniffyMemoryBlock(space="user_1042", client=Client(), instructions="Facts about the user:")
    assert asyncio.run(block.aget([ChatMessage(role="user", content="hi")])) == f"Facts about the user:\n\n{BRIEFING}"
    bare = GeniffyMemoryBlock(space="user_1042", client=Client(), instructions="")
    assert asyncio.run(bare.aget([ChatMessage(role="user", content="hi")])) == BRIEFING


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
    assert BRIEFING in asyncio.run(block.aget([ChatMessage(role="user", content="hi")]))
    asyncio.run(block.remember("hi", "hello"))
    recall, _ = geniffy_tools(space="user_1042", client=client)
    assert str(recall.call(query="q")) == CONTEXT
    assert len(client.log["briefing"]) == 1 and len(client.log["context"]) == 1 and len(client.log["session"]) == 1


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
    assert made[0].log["briefing"] == [("user_1", None, "hi"), ("user_2", None, "hi")]

    asyncio.run(twice())
    assert len(made) == 2, "a new loop gets its own"
    assert len(base._shared) == 1, "and the closed loop's client is let go"
