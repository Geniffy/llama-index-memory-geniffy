# Changelog

## 0.1.0

The first release.

- `GeniffyMemoryBlock(space=...)`: a LlamaIndex memory block that puts what is known about the user in the system
  message, each line with where it came from. Geniffy is asked once per question however many steps a run takes,
  and an agent goes on without memory when Geniffy can't be reached.
- `block.remember(message, reply)`: saves an exchange, from strings, ChatMessages or what `agent.run` returned.
- `geniffy_tools(space=...)`: `recall` and `remember` for one user.
- A space is required and can't be blank, so one user's memory is never written to another's or to your own.
