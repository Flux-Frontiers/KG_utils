# Release Notes — v0.30.2

> Released: 2026-11-17

### Fixed

- **A synthesis key goes only to its own backend.** `text_config_from_env()`
  fell back to `OPENAI_API_KEY` for every backend and to `VLLM_API_KEY` for
  every backend but Anthropic. So the default `omlx` backend sent the OpenAI
  key to the local oMLX server whenever `VLLM_API_KEY` was unset, which oMLX
  rejected with 401; AgentKG's `prune` hit this. Now `omlx` falls back to
  `VLLM_API_KEY` only, `openai` to `OPENAI_API_KEY` only, `anthropic` reads
  `ANTHROPIC_API_KEY`, and `ollama` takes `SYNTH_API_KEY` or nothing.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
