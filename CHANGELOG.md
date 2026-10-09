# Changelog

## 0.2.0

- **The briefing.** The block gives the system message the user's briefing: where things stand, what is due or was
  promised, the rules that apply, what happened, then what is known that bears on their latest message, each line
  dated. `project=` keeps it to one project, `budget_chars=` caps it, and `briefing=False` reads only what bears on
  the question, as 0.1 did. An account without the briefing switched on gets that too, and the briefing is tried
  again ten minutes later. A briefing with nothing in it yet says so, so the model never reads silence as
  permission to guess.
- **`await block.save(memory)`** after each run keeps the whole run, tool calls and their results included, in one
  memory for the conversation (the Memory's `session_id`). Saving the same run twice sends nothing the second time,
  a save that fails goes with the next one, and over 500 messages go 500 at a time. An app that keeps the history
  itself passes its messages and `session=`.
- With `accept_short_term_memory=True`, what overflows short-term history keeps its tool calls too, and goes into
  the conversation it came from.
- `remember(message, reply)` still saves one exchange, and takes `session=`.
- Needs `geniffy` 0.4.0 or later.

## 0.1.0

The first release.

- `GeniffyMemoryBlock(space=...)`: a LlamaIndex memory block that puts what is known about the user in the system
  message, each line with where it came from. Geniffy is asked once per question however many steps a run takes,
  and an agent goes on without memory when Geniffy can't be reached.
- `block.remember(message, reply)`: saves an exchange, from strings, ChatMessages or what `agent.run` returned.
- `geniffy_tools(space=...)`: `recall` and `remember` for one user.
- A space is required and can't be blank, so one user's memory is never written to another's or to your own.
