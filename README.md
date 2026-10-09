<p align="center">
  <a href="https://geniffy.com"><img src="https://geniffy.com/brand/geniffy-lockup-ink.png" alt="Geniffy" height="44"></a>
</p>

# Geniffy for LlamaIndex

Give a LlamaIndex agent a memory of each of your users. Add one memory block, and before the agent answers,
LlamaIndex puts the user's briefing into the system message: where things stand, what is due, the rules that
apply, what happened, and what is known that bears on their message. After each run, save it, tool calls
included, into one memory for the whole conversation. When nothing is known, the model is told so, and says so
instead of guessing.

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


async def chat(user_id: str, chat_id: str, message: str) -> str:
    block = GeniffyMemoryBlock(space=f"user_{user_id}")       # the user from your own sign-in
    memory = Memory.from_defaults(session_id=chat_id, memory_blocks=[block])
    reply = await agent.run(user_msg=message, memory=memory)
    await block.save(memory)                                  # the run, saved into that user's conversation
    return str(reply)
```

- **Before the agent answers**, the user's briefing goes in the system message, inside LlamaIndex's `<memory>`
  section: where things stand, what is due or was promised, the rules that apply, what happened, then what is
  known that bears on their latest message, each line dated. In a run with tools, Geniffy is asked once, not once
  per step.
- **`save(memory)`** keeps the run that just finished: what the user said, each tool the agent called and what it
  returned, and the answer. The Memory's `session_id` is the conversation, and a conversation is one memory however
  long it gets. Saving the same run twice sends nothing the second time, and a save that fails goes with the next
  one. An app that keeps the history itself passes its messages and `session=` its id instead.
- **If Geniffy can't be reached**, the agent goes on without memory, and `on_error` hears about it.

Options: `project` keeps the briefing to one project and labels what is saved with it; `briefing=False` reads only
what bears on the user's latest message; `budget_chars` is the most the briefing adds (6,000 characters unless you
say); `instructions` changes what the model is told about the memory; and `client` shares your own
`geniffy.AsyncGeniffy`. Without one, a client is made for each event loop and shared by every block. An account
without the briefing switched on is given what bears on the latest message instead, and the briefing is tried
again ten minutes later.

LlamaIndex hands a block only the messages that overflow its short-term history, which most conversations never
reach, so `save()` is how each run gets to Geniffy. To save the overflow instead, pass
`accept_short_term_memory=True` and don't call `save()`, or a run is saved twice. `remember(message, reply)` still
saves a single exchange, without the tool calls.

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
                                       # and the real geniffy client against a stand-in for the API
```

## Security

Report a vulnerability to ops@geniffy.com, not in a public issue. See the
[security policy](https://github.com/Geniffy/.github/blob/main/SECURITY.md).
