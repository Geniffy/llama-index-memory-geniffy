"""GeniffyMemoryBlock and geniffy_tools."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.tools import FunctionTool
from pydantic import ConfigDict, Field, PrivateAttr, field_validator

logger = logging.getLogger("llama_index.memory.geniffy")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what this app knows from before, each line with its date. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)
# A briefing with nothing in it yet still says so: a model reads an empty memory as permission to invent.
NOTHING_YET = "Nothing is remembered about this user yet."

_NO_SPACE = ("Geniffy needs a space: the user this is for, such as space=f\"user_{user.id}\". "
             "Pass space=None only for your own memory, never for your users' data.")

_REUSE_S = 30.0       # an agent reads its memory at every step of a run: the same question is asked of Geniffy once
_OFF_S = 600.0        # how long only what bears on the question is read after the briefing was found switched off
_PER_CALL = 500       # the most messages Geniffy takes in one call
_HELD = 256           # conversations whose failed saves are held for the next save
# What Geniffy keeps of a message, so nothing larger is sent: a turn's words, a tool call's input and a tool's output.
_TURN_CHARS, _CALL_CHARS, _RESULT_CHARS = 50_000, 2_000, 4_000


def _check_space(space: Any) -> Optional[str]:
    """A space names one user. A blank one would quietly mean your own memory, so it is refused."""
    if space is None:
        return None
    if not isinstance(space, str):
        raise TypeError(f"space must be a string (or None for your own memory), not {type(space).__name__}")
    if not space.strip():
        raise ValueError("space is blank: it would be your own memory, not a user's. " + _NO_SPACE)
    return space


# One AsyncGeniffy per event loop when no client is passed: a client's connections belong to the loop that
# opened them, so an app on one loop (a server) keeps one, and a script that calls asyncio.run per job gets
# one per run, the closed loops' clients let go.
_shared: Dict[int, Tuple[asyncio.AbstractEventLoop, Any]] = {}


def _default_client() -> Any:
    loop = asyncio.get_running_loop()
    held = _shared.get(id(loop))
    if held is None or held[0] is not loop:
        for key, (old, _) in list(_shared.items()):
            if old.is_closed():
                _shared.pop(key, None)
        from geniffy import AsyncGeniffy
        held = _shared[id(loop)] = (loop, _made(AsyncGeniffy))
    return held[1]


def _bound(client: Any, space: Optional[str]) -> Any:
    """The client, pointed at one user's space (or your own memory, for None)."""
    client = client if client is not None else _default_client()
    return client if space is None else client.space(space)


async def _done(value: Any) -> Any:
    """An AsyncGeniffy's answer, or a Geniffy's: a sync client works too, but holds up the event loop."""
    return await value if inspect.isawaitable(value) else value


def _text(message: Any) -> str:
    """The words of a string, a ChatMessage, or what agent.run returned (an AgentOutput, words in .response)."""
    if isinstance(message, str):
        return message.strip()
    if isinstance(getattr(message, "response", None), ChatMessage):
        message = message.response
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    blocks = getattr(message, "blocks", None) or []
    return "\n".join(b.text for b in blocks if isinstance(getattr(b, "text", None), str)).strip()


def _role(message: Any) -> str:
    return str(getattr(message.role, "value", message.role))


def _last(messages: Optional[List[ChatMessage]], role: str) -> str:
    for message in reversed(list(messages or [])):
        if _role(message) == role:
            return _text(message)
    return ""


def _turn_start(messages: List[ChatMessage]) -> int:
    """Where the run that just finished starts: just after the last answer before the user's last message, so two
    messages sent at once are both in it."""
    users = [i for i, m in enumerate(messages) if _role(m) == "user"]
    if not users:
        return 0
    for i in range(users[-1] - 1, -1, -1):
        if _role(messages[i]) in ("assistant", "chatbot", "model"):
            return i + 1
    return 0


def _call(name: Any, args: Any, call_id: Any) -> Dict[str, Any]:
    try:
        shown = args if isinstance(args, str) else json.dumps(args, default=str)
    except (TypeError, ValueError):
        shown = str(args)
    out: Dict[str, Any] = {"name": str(name or "tool")[:200],
                           "args": args if len(shown) <= _CALL_CHARS else shown[:_CALL_CHARS]}
    if call_id:
        out["id"] = str(call_id)[:200]
    return out


def _calls(message: ChatMessage) -> List[Dict[str, Any]]:
    """The tools an assistant message called: as blocks (LlamaIndex 0.14), or as the provider held them."""
    out = [_call(b.tool_name, b.tool_kwargs, b.tool_call_id) for b in getattr(message, "blocks", None) or []
           if getattr(b, "block_type", None) == "tool_call"]
    if not out:
        for c in (message.additional_kwargs or {}).get("tool_calls") or []:
            get = c.get if isinstance(c, dict) else (lambda k, c=c: getattr(c, k, None))
            fn = get("function")
            fget = fn.get if isinstance(fn, dict) else (lambda k, fn=fn: getattr(fn, k, None))
            out.append(_call(fget("name"), fget("arguments"), get("id")))
    return out[:100]


def _as_sent(message: ChatMessage) -> Optional[Dict[str, Any]]:
    """A ChatMessage as Geniffy reads a conversation: what the user said, what the agent answered with the tools it
    called, and what each tool returned. System messages are the app's instructions, and are left out."""
    role = _role(message)
    extra = message.additional_kwargs or {}
    if role in ("tool", "function"):
        out: Dict[str, Any] = {"role": "tool", "content": _text(message)[:_RESULT_CHARS]}
        name = extra.get("name") or extra.get("tool_name")
        if name:
            out["name"] = str(name)[:200]
        if extra.get("tool_call_id"):
            out["tool_call_id"] = str(extra["tool_call_id"])[:200]
        return out
    if role == "user":
        return {"role": "user", "content": _text(message)[:_TURN_CHARS]}
    if role in ("assistant", "chatbot", "model"):
        out = {"role": "assistant", "content": _text(message)[:_TURN_CHARS]}
        calls = _calls(message)
        if calls:
            out["tool_calls"] = calls
        return out
    return None


def _sent(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    out = [s for s in map(_as_sent, messages) if s is not None]
    return [s for s in out if s.get("content") or s.get("tool_calls") or s["role"] == "tool"]


def _title(sent: List[Dict[str, Any]]) -> Optional[str]:
    """What to call a saved conversation in the Geniffy app: the start of its first question, on one line."""
    for s in sent:
        words = " ".join(str(s.get("content") or "").split()) if s["role"] == "user" else ""
        if words:
            return words if len(words) <= 80 else words[:80].rsplit(" ", 1)[0] + "..."
    return None


class GeniffyMemoryBlock(BaseMemoryBlock[str]):
    """The user's briefing, for LlamaIndex's Memory to put in the system message: where things stand, what is due,
    the rules that apply, what happened, then what is known that bears on their latest message, each line dated.
    When nothing is known, the model is told so, and says so instead of guessing. After each run, save() keeps the
    run, tool calls included, in one memory for the whole conversation.

        block = GeniffyMemoryBlock(space=f"user_{user.id}")
        memory = Memory.from_defaults(session_id=chat.id, memory_blocks=[block])
        reply = await agent.run(user_msg=message, memory=memory)
        await block.save(memory)

    space         the user this is for (required; None is your own memory, never your users' data)
    project       the project the agent works on: the briefing keeps to it, and what is saved is labelled with it
    briefing      open with the briefing (default True); False reads only what is known that bears on the
                  user's latest message. An account without the briefing reads that, either way
    budget_chars  the most the briefing adds to the system message, in characters (500 to 40,000; default 6,000)
    client        a geniffy.AsyncGeniffy to share (default: one per event loop, reading GENIFFY_API_KEY)
    instructions  what the model is told about the memory it is given
    on_error      called when Geniffy can't be reached; the agent goes on without memory
    accept_short_term_memory
                  also save the messages Memory moves out of its short-term history (default False: save each
                  run with save(), so every one reaches Geniffy, not only long sessions; use one or the other, or
                  a run is saved twice)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "geniffy"
    description: Optional[str] = ("The user's briefing: where things stand, what is due, the rules that apply, what "
                                  "happened, and what is known that bears on their latest message")
    accept_short_term_memory: bool = False
    space: Optional[str] = Field(default=None, description="The user this memory is for; None for your own memory")
    project: Optional[str] = Field(default=None, description="The project: the briefing keeps to it, and what is "
                                                             "saved is labelled with it")
    briefing: bool = True
    budget_chars: int = Field(default=6000, ge=500, le=40_000)
    instructions: str = DEFAULT_INSTRUCTIONS
    client: Any = None
    on_error: Optional[Callable[[Exception], None]] = None
    _recent: Optional[Tuple[str, float, str]] = PrivateAttr(default=None)
    _off_until: float = PrivateAttr(default=0.0)
    _saved: "OrderedDict[str, Tuple[int, str]]" = PrivateAttr(default_factory=OrderedDict)
    _held: "OrderedDict[str, List[Dict[str, Any]]]" = PrivateAttr(default_factory=OrderedDict)

    def __init__(self, **data: Any):
        if "space" not in data:
            raise TypeError(_NO_SPACE)
        super().__init__(**data)

    @field_validator("space", mode="before")
    @classmethod
    def _one_user(cls, space: Any) -> Optional[str]:
        return _check_space(space)

    def _memory(self) -> Any:
        return _bound(self.client, self.space)

    def _failed(self, error: Exception) -> None:
        if self.on_error is not None:
            self.on_error(error)
        else:
            logger.warning("Geniffy: %s", error)

    # ── before the agent answers: the briefing ─────────────────────────────────────────────────────────────
    async def _read(self, question: str) -> str:
        mem = self._memory()
        if self.briefing and time.monotonic() >= self._off_until:
            try:
                text = await _done(mem.briefing(project=self.project, cue=question, budget_chars=self.budget_chars))
                return str(text or "") or NOTHING_YET
            except Exception as e:  # noqa: BLE001
                # not switched on for this key (404), or an API from before it (405): what bears on the question
                # is read instead, and the briefing is tried again in ten minutes
                if getattr(e, "status", None) not in (404, 405):
                    raise
                self._off_until = time.monotonic() + _OFF_S
        return str(await _done(mem.context(question))) if question else ""

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        question = _last(messages, "user")
        held = self._recent
        if held and held[0] == question and time.monotonic() - held[1] < _REUSE_S:
            context = held[2]
        else:
            try:
                context = await self._read(question)
            except Exception as e:  # noqa: BLE001 - the agent goes on without memory, and the rest of
                self._failed(e)     # the run doesn't wait on Geniffy again at every step
                context = ""
            self._recent = (question, time.monotonic(), context)
        if not context:
            return ""
        return f"{self.instructions}\n\n{context}" if self.instructions else context

    # ── after the run: the conversation, saved as it goes ──────────────────────────────────────────────────
    async def _send(self, sent: List[Dict[str, Any]], session: Optional[str]) -> None:
        """Into one memory for the conversation, 500 messages a call. A conversation's save that fails is held, and
        goes with its next save; without a conversation, each save is a memory of its own."""
        batch = (self._held.pop(session, []) if session else []) + sent
        if not batch:
            return
        self._recent = None  # what was just said can bear on the next question
        name, done = session or f"run-{uuid.uuid4().hex}", 0
        labels = {"project": self.project} if self.project else None
        try:
            for i in range(0, len(batch), _PER_CALL):
                part = batch[i:i + _PER_CALL]
                await _done(self._memory().memories.add(messages=part, session=name, title=_title(batch),
                                                        labels=labels))
                done = i + len(part)
        except Exception as e:  # noqa: BLE001
            if session:
                self._held[session] = batch[done:]
                while len(self._held) > _HELD:
                    self._held.popitem(last=False)
            self._failed(e)

    async def save(self, memory: Any, *, session: Optional[str] = None) -> None:
        """Save the run that just finished: what the user said, each tool the agent called and what it returned, and
        the answer, into one memory for the conversation. Call it after each agent.run with the run's Memory (the
        conversation is its session_id), or with the messages your app keeps and session= its id.

            reply = await agent.run(user_msg=message, memory=memory)
            await block.save(memory)

        Saving the same run twice sends nothing the second time."""
        if hasattr(memory, "aget_all"):
            messages = list(await memory.aget_all())
            session = session or (str(getattr(memory, "session_id", "") or "") or None)
        else:
            messages = list(memory or [])
        sent = _sent(messages[_turn_start(messages):])
        if not sent:
            return
        if session:
            mark = (len(messages), json.dumps(sent[-1], sort_keys=True, default=str))
            if self._saved.get(session) == mark:
                return
            self._saved[session] = mark
            while len(self._saved) > _HELD:
                self._saved.popitem(last=False)
        await self._send(sent, session)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """With accept_short_term_memory=True, what Memory moves out of its short-term history, tool calls and results
        included, into the conversation it came from."""
        session = next((m.additional_kwargs.get("session_id") for m in messages
                        if (m.additional_kwargs or {}).get("session_id")), None)
        await self._send(_sent(messages), str(session) if session else None)

    async def remember(self, user_message: Any, reply: Any, *, session: Optional[str] = None) -> None:
        """Save one exchange: what the user said and the answer (a string, a ChatMessage, or what agent.run
        returned), without the tools the agent called; save() keeps those too. The user's words become facts about
        the user."""
        asked, answer = _text(user_message), _text(reply)
        if asked and answer:
            await self._send([{"role": "user", "content": asked[:_TURN_CHARS]},
                              {"role": "assistant", "content": answer[:_TURN_CHARS]}], session)


def geniffy_tools(*, space: Optional[str] = ..., client: Any = None) -> List[FunctionTool]:  # type: ignore[assignment]
    """[recall, remember] for one user, for an agent that decides when to look something up or save it. Make
    them per user (per request), so the model never chooses whose memory it reads.

        agent = FunctionAgent(tools=geniffy_tools(space=f"user_{user.id}"), llm=llm)
    """
    if space is ...:
        raise TypeError(_NO_SPACE)
    space = _check_space(space)

    async def recall(query: str) -> str:
        """Look up what is known about the user: their preferences, people, plans and what they said before, each
        line with where it came from. When nothing is known, it says so."""
        return str(await _done(_bound(client, space).context(query)))

    async def remember(text: str) -> str:
        """Save something the user asked you to remember, in their words. Never passwords, keys or other
        secrets."""
        await _done(_bound(client, space).memories.add(text))
        return "Saved."

    return [FunctionTool.from_defaults(async_fn=recall, name="recall"),
            FunctionTool.from_defaults(async_fn=remember, name="remember")]


def _made(cls: Any) -> Any:
    """A client of the geniffy SDK, named after this package, so the Requests page in the Geniffy app shows
    which integration made each call. A geniffy before 0.2.0 has no name to give, and makes a plain client."""
    from . import __version__
    try:
        return cls(integration=f"llama-index-memory-geniffy/{__version__}")
    except TypeError:
        return cls()
