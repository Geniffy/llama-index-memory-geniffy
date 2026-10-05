"""GeniffyMemoryBlock and geniffy_tools."""
from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from llama_index.core.llms import ChatMessage
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.tools import FunctionTool
from pydantic import ConfigDict, Field, PrivateAttr, field_validator

logger = logging.getLogger("llama_index.memory.geniffy")

DEFAULT_INSTRUCTIONS = (
    "What follows is this user's memory: what they told this app before, each line with where it came from. "
    "Use it when it helps and don't recite it. If it doesn't cover something, say so instead of guessing."
)

_NO_SPACE = ("Geniffy needs a space: the user this is for, such as space=f\"user_{user.id}\". "
             "Pass space=None only for your own memory, never for your users' data.")

_REUSE_S = 30.0  # an agent reads its memory at every step of a run: the same question is asked of Geniffy once


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


class GeniffyMemoryBlock(BaseMemoryBlock[str]):
    """What Geniffy knows about one of your users that bears on their latest message, for LlamaIndex's Memory
    to put in the system message, each line with where it came from. When nothing is known, the model is told
    so, and says so instead of guessing.

        block = GeniffyMemoryBlock(space=f"user_{user.id}")
        memory = Memory.from_defaults(session_id=f"user_{user.id}", memory_blocks=[block])
        reply = await agent.run(user_msg=message, memory=memory)
        await block.remember(message, reply)

    space         the user this is for (required; None is your own memory, never your users' data)
    client        a geniffy.AsyncGeniffy to share (default: one per event loop, reading GENIFFY_API_KEY)
    instructions  what the model is told about the memory it is given
    on_error      called when Geniffy can't be reached; the agent goes on without memory
    accept_short_term_memory
                  also save the messages Memory moves out of its short-term history (default False: save each
                  exchange with remember(), so every one reaches Geniffy, not only long sessions; use one or
                  the other, or an exchange is saved twice)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "geniffy"
    description: Optional[str] = "What Geniffy knows about the user that bears on their latest message"
    accept_short_term_memory: bool = False
    space: Optional[str] = Field(default=None, description="The user this memory is for; None for your own memory")
    instructions: str = DEFAULT_INSTRUCTIONS
    client: Any = None
    on_error: Optional[Callable[[Exception], None]] = None
    _recent: Optional[Tuple[str, float, str]] = PrivateAttr(default=None)

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

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        question = _last(messages, "user")
        if not question:
            return ""
        held = self._recent
        if held and held[0] == question and time.monotonic() - held[1] < _REUSE_S:
            context = held[2]
        else:
            try:
                context = str(await _done(self._memory().context(question)))
            except Exception as e:  # noqa: BLE001 - the agent goes on without memory, and the rest of
                self._failed(e)     # the run doesn't wait on Geniffy again at every step
                context = ""
            self._recent = (question, time.monotonic(), context)
        if not context:
            return ""
        return f"{self.instructions}\n\n{context}" if self.instructions else context

    async def _save(self, turns: List[Dict[str, str]]) -> None:
        self._recent = None  # what was just said can bear on the next question
        try:
            await _done(self._memory().memories.add(messages=turns))
        except Exception as e:  # noqa: BLE001
            self._failed(e)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Save the user's and the assistant's messages among these (tool calls and results are left out)."""
        turns = [{"role": _role(m), "content": _text(m)} for m in messages]
        turns = [t for t in turns if t["role"] in ("user", "assistant") and t["content"]]
        if turns:
            await self._save(turns)

    async def remember(self, user_message: Any, reply: Any) -> None:
        """Save one exchange: what the user said and the answer (a string, a ChatMessage, or what agent.run
        returned). The user's words become facts about the user."""
        asked, answer = _text(user_message), _text(reply)
        if asked and answer:
            await self._save([{"role": "user", "content": asked}, {"role": "assistant", "content": answer}])


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
