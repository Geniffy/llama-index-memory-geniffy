"""Geniffy for LlamaIndex: a memory of each of your users, for any LlamaIndex agent.

    GeniffyMemoryBlock(space=...)   a memory block: LlamaIndex puts what is known about the user that bears on
                                    their latest message into the system message
    block.remember(user, reply)     save the exchange after a run
    geniffy_tools(space=...)        recall and remember, as tools the agent calls itself

`space` names the user, and is required, so one user's memory is never written to another's; `space=None` is your
own memory, never your users' data.
"""
from llama_index.memory.geniffy.base import GeniffyMemoryBlock, geniffy_tools

__all__ = ["GeniffyMemoryBlock", "geniffy_tools"]
__version__ = "0.1.0"
