<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for LlamaIndex

Give a LlamaIndex agent a memory of each of your users. Add one memory block, and before the agent answers,
LlamaIndex puts what is known about the user into the system message, each line with where it came from.
After the run, save the exchange. When nothing is known, the model is told so, and says so instead of
guessing.

[![CI](https://github.com/Geniffy/llama-index-memory-geniffy/actions/workflows/ci.yml/badge.svg)](https://github.com/Geniffy/llama-index-memory-geniffy/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llama-index-memory-geniffy)](https://pypi.org/project/llama-index-memory-geniffy/)
[![Docs](https://img.shields.io/badge/docs-docs.geniffy.com-1A1814)](https://docs.geniffy.com/integrations/llamaindex)

```bash
pip install llama-index-memory-geniffy
```

Set `GENIFFY_API_KEY` from **API keys** in the [Geniffy app](https://geniffy.com/app). Keep it on your server.

## The memory block

```python
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.llms.anthropic import Anthropic
from llama_index.memory.geniffy import GeniffyMemoryBlock

agent = FunctionAgent(llm=Anthropic(model="claude-opus-5-5"), system_prompt="You are a helpful assistant.")


async def chat(user_id: str, message: str) -> str:
    block = GeniffyMemoryBlock(space=f"user_{user_id}")       # the user from your own sign-in
    memory = Memory.from_defaults(session_id=f"user_{user_id}", memory_blocks=[block])
    reply = await agent.run(user_msg=message, memory=memory)
    await block.remember(message, reply)                      # the exchange, saved to that user's space
    return str(reply)
```

- **Before the agent answers**, what is known that bears on the user's latest message goes in the system
  message, inside LlamaIndex's `<memory>` section. In a run with tools, Geniffy is asked once, not once per step.
- **`remember()`** saves the exchange. LlamaIndex only hands a block the messages that overflow its short-term
  history, which most conversations never reach, so each exchange is saved this way instead. To save the
  overflow too, pass `accept_short_term_memory=True` and don't call `remember()`, or an exchange is saved twice.
- **If Geniffy can't be reached**, the agent goes on without memory, and `on_error` hears about it.

Options: `instructions` to change what the model is told about the memory, and `client` to share your own
`geniffy.AsyncGeniffy`. Without one, a client is made for each event loop and shared by every block.

## Tools

To let the agent decide when to look something up or save it:

```python
from llama_index.memory.geniffy import geniffy_tools

agent = FunctionAgent(llm=llm, tools=geniffy_tools(space=f"user_{user_id}"))   # recall and remember
```

Make the tools per user (per request), so the model never sees or chooses whose memory it reads.

## One space per user

`space` is required: the user this is for. Each space is a memory of its own, and nothing else can read it.
`space=None` is your own memory, never your users' data, and a blank space is refused, so a user with no id
can't end up in it. When a user deletes their account, forget them with `AsyncGeniffy().forget_space(...)`
from the [geniffy](https://pypi.org/project/geniffy/) SDK.

## Develop

```bash
pip install -e ".[test]" && pytest     # through LlamaIndex's own FunctionAgent and Memory, with a scripted LLM
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
