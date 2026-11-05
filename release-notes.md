# Release Notes — v0.30.0

> Released: 2026-11-05

### Added

- **An Anthropic text backend.** `TextBackend.ANTHROPIC` (`SYNTH_BACKEND=anthropic`,
  or `backend="anthropic"` per request) calls Claude through the `anthropic` SDK,
  now a dependency of the `synthesis` extras. System messages become the
  `system` parameter; no temperature is sent; effort is `low`, with a retry
  without it for models that refuse it; `max_tokens` is raised to at least
  4096 to leave room for thinking; the current models get the server-side
  refusal fallback. The default model is `claude-opus-5-5`, and the key is
  `ANTHROPIC_API_KEY` (an OpenAI key is never passed to it).

### Fixed

- **OpenAI gpt-5 and later answer again.** The OpenAI backend sent
  `max_tokens` and `temperature=0.7`; gpt-5, gpt-5.x, gpt-6, o3 and
  `chat-latest` reject one or both with a 400, and the error was swallowed,
  so only the GPT-4 family answered. It now sends `max_completion_tokens`,
  and retries without the temperature when a model refuses it, remembering
  that model. A failed completion is logged as a warning.
- **The OpenAI model list holds only chat models.** `list_models` on OpenAI
  drops audio, image, realtime, transcription, tts, embedding, moderation,
  search, codex and `-pro` models, none of which answer through chat
  completions.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
