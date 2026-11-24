# kg_utils.synthesis — Unified Synthesis Backends

`kg_utils.synthesis` provides a single, consistent interface for text generation
across multiple local and cloud backends.  The oMLX, Ollama and OpenAI text
backends use the **OpenAI wire protocol** (`/v1/chat/completions`); the Anthropic backend
calls Claude through the official `anthropic` SDK. Switching between them requires only
an environment variable change — no code changes.

Image generation moved to [imagewright](https://github.com/Flux-Frontiers/imagewright)
in 0.32.0. `TextSynthesizer.rewrite_for_image()` stays here: it turns a passage
into an image prompt for whatever generates the image.

---

## Installation

```bash
# Text synthesis (oMLX, Ollama, OpenAI, Anthropic)
pip install 'kgmodule-utils[synthesis]'
```

**Optional extra dependencies:**

| Extra | Adds |
|---|---|
| `synthesis` | `anthropic`, `httpx`, `openai`, `pillow` |

---

## Quick Start

```python
from kg_utils.synthesis import text_synthesizer_from_env

text = text_synthesizer_from_env()   # reads SYNTH_BACKEND / SYNTH_ENDPOINT / …

# Grounded RAG answer from corpus hits
answer = text.synthesize_rag(query, hits)

# Rewrite a historical passage into an image prompt
prompt, err = text.rewrite_for_image(corpus_passage)
```

---

## Environment Variables

### Text synthesis

| Variable | Default | Description |
|---|---|---|
| `SYNTH_BACKEND` | `omlx` | Backend: `omlx` \| `ollama` \| `openai` \| `anthropic` |
| `SYNTH_ENDPOINT` | *(backend default)* | Override base URL |
| `SYNTH_API_KEY` | `""` | Bearer token for the selected backend. Unset, `omlx` falls back to `VLLM_API_KEY` and `openai` to `OPENAI_API_KEY`; `anthropic` reads `ANTHROPIC_API_KEY`. No backend gets another backend's key |
| `SYNTH_MODEL` | *(backend default)* | Model-id override |

**Legacy aliases** (honoured as fallbacks, no migration needed):

| Legacy var | Maps to |
|---|---|
| `VLLM_ENDPOINT_URL` | `SYNTH_ENDPOINT` |
| `VLLM_API_KEY` | `SYNTH_API_KEY`, for the `omlx` backend only |
| `VLLM_MODEL` | `SYNTH_MODEL` |

---

## Backend Defaults

### Text backends

| Backend | Default endpoint | Default model |
|---|---|---|
| `omlx` | `http://localhost:8080/v1` | `Qwen3-4B-Instruct-2507-MLX-8bit` |
| `ollama` | `http://localhost:11434/v1` | `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `anthropic` | *(SDK default)* | `claude-opus-5-5` |

---

## API Reference

### `TextBackend` (enum)

```python
class TextBackend(str, Enum):
    OMLX   = "omlx"    # local oMLX / vLLM — thinking suppressed via extra_body
    OLLAMA = "ollama"  # local Ollama — no api_key required
    OPENAI = "openai"  # OpenAI cloud — requires OPENAI_API_KEY / SYNTH_API_KEY
    ANTHROPIC = "anthropic"  # Claude API — anthropic SDK; ANTHROPIC_API_KEY or an `ant` profile
```

**OpenAI.** Requests send `max_completion_tokens`, which every OpenAI chat model
accepts (gpt-5 and later reject `max_tokens`). A model that refuses the requested
temperature is asked again without it, and later calls for that model leave it out.
`list_models()` returns only the models chat completions can answer with.

**Anthropic.** System messages become the `system` parameter. No temperature is sent
(current Claude models reject sampling parameters). Effort is `low`, retried without it
for a model that refuses it, and `max_tokens` is raised to at least 4096 because thinking
counts against it. The current models (Fable 5.1, Opus 5.5, Opus 5, Sonnet 5.5) get the
server-side refusal fallback. A refusal, an API error or missing credentials is logged
and returns `None`.

### `TextConfig` (dataclass)

```python
@dataclass
class TextConfig:
    backend:           TextBackend = TextBackend.OMLX
    endpoint:          str         = ""       # empty → backend default
    api_key:           str         = ""       # empty → not required for omlx/ollama
    model:             str         = ""       # empty → backend default
    max_tokens:        int         = 2048
    suppress_thinking: bool        = True     # strip <think> blocks; pass extra_body to oMLX
```

Helper methods:

| Method | Returns |
|---|---|
| `resolved_endpoint()` | Effective base URL (override or backend default) |
| `resolved_model()` | Effective model ID (override or backend default) |

Factory: `text_config_from_env() -> TextConfig`

### `TextSynthesizer`

```python
synth = TextSynthesizer(config: TextConfig)
synth = text_synthesizer_from_env()   # convenience factory
```

| Method | Returns | Description |
|---|---|---|
| `list_models()` | `list[str]` | Model IDs available at the endpoint; `[]` on failure |
| `synthesize_rag(query, snippets, *, model, max_k, system)` | `str \| None` | Grounded answer from corpus hits |
| `rewrite_for_image(corpus_text, *, model, system)` | `tuple[str, str \| None]` | `(image_prompt, error)`; `system` defaults to `IMAGE_REWRITE_SYSTEM` (historical prose), or pass `IMAGE_REWRITE_TECHNICAL_SYSTEM` for code, records, memory or graph data |

#### `synthesize_rag` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | — | Natural-language question |
| `snippets` | `list[dict]` | — | Hit dicts — must contain `content` key |
| `model` | `str \| None` | `None` | Per-call model override |
| `max_k` | `int` | `12` | Maximum snippets passed to the context window |
| `system` | `str \| None` | `None` | Override default RAG system prompt |

Each snippet dict may carry optional metadata keys that appear in the context header:
`genre`, `author`, `title`, `name`, `kg_kind`.  Snippets whose `content` is empty or
whitespace-only are silently skipped.

Returns `None` when no snippets carry content or when the backend returns no response.

#### `rewrite_for_image` system prompts

`kg_utils.synthesis` exports two system prompts for the rewrite:

| Name | For | Asks for |
|---|---|---|
| `IMAGE_REWRITE_SYSTEM` (default) | Historical prose: diaries, books | The scene as the passage leaves it, with people in period dress |
| `IMAGE_REWRITE_TECHNICAL_SYSTEM` | Non-prose text: source code, structured records, agent memory, graph data such as neurons and pathways | A clear technical or scientific illustration, with no invented people or period scenes |

```python
from kg_utils.synthesis import IMAGE_REWRITE_TECHNICAL_SYSTEM

prompt, err = text.rewrite_for_image(code_snippet, system=IMAGE_REWRITE_TECHNICAL_SYSTEM)
```

Through a LAN worker, pass the same string to `WorkerClient.rewrite(text, system=...)`.
It travels as an optional `system` field of the `rewrite` op. A worker on a
kgmodule-utils older than 0.31.0 ignores the field and uses the historical prompt.

#### `rewrite_for_image` return value

Returns `(prompt, error)`.  On success `error` is `None`.  On failure `prompt` falls
back to the original `corpus_text` and `error` carries a short explanation string.

---

## Usage Patterns

### Switch backends without code changes

```bash
# Use Ollama instead of oMLX
export SYNTH_BACKEND=ollama

# Use OpenAI GPT-4o
export SYNTH_BACKEND=openai
export OPENAI_API_KEY=sk-...
export SYNTH_MODEL=gpt-4o
```

### Use explicit config in code

```python
from kg_utils.synthesis import TextBackend, TextConfig, TextSynthesizer

# Explicit Ollama config
text = TextSynthesizer(TextConfig(
    backend=TextBackend.OLLAMA,
    model="llama3:8b",
))
```

### RAG pipeline integration

```python
from kg_utils.synthesis import text_synthesizer_from_env

text = text_synthesizer_from_env()

# Corpus hits from KGRAG / DocKG
hits = kgrag.query("great fire of London", k=8).hits

# Synthesise a grounded answer
answer = text.synthesize_rag(
    "What does Pepys say about the Great Fire?",
    [{"content": h.text, "author": h.author, "title": h.title} for h in hits],
    max_k=8,
)

# Rewrite the top hit into an image prompt for imagewright
prompt, _ = text.rewrite_for_image(hits[0].text)
```

---

## Notes

- **oMLX thinking suppression** — `TextSynthesizer` automatically passes
  `extra_body={"think": False, "chat_template_kwargs": {"enable_thinking": False}}`
  to oMLX endpoints and strips any residual `<think>…</think>` blocks from the
  response.  This is a no-op for Ollama and OpenAI.

- **Ollama authentication** — Ollama does not require an API key.  The synthesizer
  passes `"not-needed"` internally when `api_key` is empty, satisfying the
  `openai` client's non-optional parameter.

- **`list_models()` on Ollama** — returns all locally-pulled models (not just the
  configured one), so the count may be large.  Use the configured model ID
  (`synth._cfg.resolved_model()`) rather than relying on list position.

- **Error handling** — `synthesize_rag()` returns `None` on any backend exception.
  `rewrite_for_image()` returns `(original_text, error_message)` so callers always
  have a usable prompt string even when synthesis fails.  `list_models()` returns `[]`
  on failure.  No exceptions are propagated to the caller.
