"""Geniffy for LlamaIndex: a memory of each of your users, for any LlamaIndex agent.

    GeniffyMemoryBlock(space=...)   a memory block: LlamaIndex puts the user's briefing into the system message
    await block.save(memory)        after each run: the run, tool calls included, into one memory for the
                                    conversation
    geniffy_tools(space=...)        recall and remember, as tools the agent calls itself

`space` names the user, and is required, so one user's memory is never written to another's; `space=None` is your
own memory, never your users' data.
"""
from llama_index.memory.geniffy.base import GeniffyMemoryBlock, geniffy_tools

__all__ = ["GeniffyMemoryBlock", "geniffy_tools"]
__version__ = "0.2.0"
