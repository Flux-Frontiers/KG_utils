# kg_utils.synthesis — Unified Synthesis Backends

`kg_utils.synthesis` provides a single, consistent interface for text generation and image
generation across multiple local and cloud backends.  All three text backends use the
**OpenAI wire protocol** (`/v1/chat/completions`), so switching between them requires only
an environment variable change — no code changes.

---

## Installation

```bash
# Text synthesis only (oMLX, Ollama, OpenAI text)
pip install 'kgmodule-utils[synthesis]'

# + local mflux image generation (Apple Silicon)
pip install 'kgmodule-utils[synthesis-mflux]'
```

**Optional extra dependencies:**

| Extra | Adds |
|---|---|
| `synthesis` | `httpx`, `openai`, `pillow` |
| `synthesis-mflux` | all of `synthesis` + `mflux` |

---

## Quick Start

```python
from kg_utils.synthesis import text_synthesizer_from_env, image_synthesizer_from_env

text  = text_synthesizer_from_env()   # reads SYNTH_BACKEND / SYNTH_ENDPOINT / …
image = image_synthesizer_from_env()  # reads IMAGE_BACKEND / IMAGE_ENDPOINT / …

# Grounded RAG answer from corpus hits
answer = text.synthesize_rag(query, hits)

# Rewrite a historical passage into an image prompt
prompt, err = text.rewrite_for_image(corpus_passage)

# Generate and return as base64 PNG
b64 = image.generate_b64(prompt, aspect_ratio="3:2")
```

---

## Environment Variables

### Text synthesis

| Variable | Default | Description |
|---|---|---|
| `SYNTH_BACKEND` | `omlx` | Backend: `omlx` \| `ollama` \| `openai` |
| `SYNTH_ENDPOINT` | *(backend default)* | Override base URL |
| `SYNTH_API_KEY` | `""` | Bearer token / OpenAI key |
| `SYNTH_MODEL` | *(backend default)* | Model-id override |

**Legacy aliases** (honoured as fallbacks, no migration needed):

| Legacy var | Maps to |
|---|---|
| `VLLM_ENDPOINT_URL` | `SYNTH_ENDPOINT` |
| `VLLM_API_KEY` | `SYNTH_API_KEY` |
| `VLLM_MODEL` | `SYNTH_MODEL` |

### Image synthesis

| Variable | Default | Description |
|---|---|---|
| `IMAGE_BACKEND` | `mflux-serve` | Backend: `mflux-local` \| `mflux-serve` \| `openai` |
| `IMAGE_ENDPOINT` | *(backend default)* | mflux-serve base URL |
| `IMAGE_API_KEY` | `""` | OpenAI API key (also reads `OPENAI_API_KEY`) |
| `IMAGE_MODEL` | *(backend default)* | Model-id override |
| `IMAGE_STEPS` | `4` | Inference steps (mflux backends only) |

**Legacy alias:**

| Legacy var | Maps to |
|---|---|
| `GUTENKG_IMAGE_MODEL` | `IMAGE_MODEL` |

---

## Backend Defaults

### Text backends

| Backend | Default endpoint | Default model |
|---|---|---|
| `omlx` | `http://localhost:8080/v1` | `Qwen3-4B-Instruct-2507-MLX-8bit` |
| `ollama` | `http://localhost:11434/v1` | `hf.co/unsloth/Qwen3-4B-Instruct-2507-GGUF:Q8_0` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |

### Image backends

| Backend | Default server / model |
|---|---|
| `mflux-local` | HF repo `mlx-community/flux2-klein-4b-4bit` (loaded in-process) |
| `mflux-serve` | `http://localhost:8090` / `flux2-klein-4b` |
| `openai` | DALL-E 3 (`dall-e-3`) |

---

## API Reference

### `TextBackend` (enum)

```python
class TextBackend(str, Enum):
    OMLX   = "omlx"    # local oMLX / vLLM — thinking suppressed via extra_body
    OLLAMA = "ollama"  # local Ollama — no api_key required
    OPENAI = "openai"  # OpenAI cloud — requires OPENAI_API_KEY / SYNTH_API_KEY
```

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
| `rewrite_for_image(corpus_text, *, model)` | `tuple[str, str \| None]` | `(image_prompt, error)` |

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

#### `rewrite_for_image` return value

Returns `(prompt, error)`.  On success `error` is `None`.  On failure `prompt` falls
back to the original `corpus_text` and `error` carries a short explanation string.

---

### `ImageBackend` (enum)

```python
class ImageBackend(str, Enum):
    MFLUX_LOCAL = "mflux-local"  # in-process Flux2Klein (Apple Silicon)
    MFLUX_SERVE = "mflux-serve"  # HTTP to a running mflux-serve instance
    OPENAI      = "openai"       # DALL-E 3
```

### `ImageConfig` (dataclass)

```python
@dataclass
class ImageConfig:
    backend:    ImageBackend = ImageBackend.MFLUX_SERVE
    server_url: str          = ""   # empty → backend default
    api_key:    str          = ""   # empty → read OPENAI_API_KEY at call time
    model:      str          = ""   # empty → backend default
    steps:      int          = 4    # inference steps (mflux only)
```

Helper methods:

| Method | Returns |
|---|---|
| `resolved_server_url()` | Effective mflux-serve URL |
| `resolved_model()` | Effective model ID |

Factory: `image_config_from_env() -> ImageConfig`

### `ImageSynthesizer`

```python
synth = ImageSynthesizer(config: ImageConfig)
synth = image_synthesizer_from_env()   # convenience factory
```

| Method | Returns | Description |
|---|---|---|
| `generate(prompt, *, aspect_ratio, seed, steps, model)` | `PIL.Image` | Generate and return PIL Image |
| `generate_b64(prompt, *, aspect_ratio, seed, steps, model)` | `str` | Base64-encoded PNG string |

#### Shared keyword arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `aspect_ratio` | `str` | `"3:2"` | One of `1:1`, `3:2`, `2:3`, `16:9`, `9:16`, `4:3`, `3:4` |
| `seed` | `int \| None` | `None` | Reproducibility seed; random int if omitted |
| `steps` | `int \| None` | `None` | Inference steps override (mflux only; ignored for DALL-E) |
| `model` | `str \| None` | `None` | Per-call model override |

#### DALL-E 3 size mapping

DALL-E 3 accepts only three resolutions.  Aspect ratios are mapped to the nearest fit:

| Requested ratio | DALL-E 3 size |
|---|---|
| `1:1` | `1024x1024` |
| `3:2`, `16:9`, `4:3` | `1792x1024` |
| `2:3`, `9:16`, `3:4` | `1024x1792` |

#### mflux pixel dimensions

| Ratio | Width | Height |
|---|---|---|
| `1:1` | 1024 | 1024 |
| `3:2` | 1536 | 1024 |
| `2:3` | 1024 | 1536 |
| `16:9` | 1536 | 864 |
| `9:16` | 864 | 1536 |
| `4:3` | 1365 | 1024 |
| `3:4` | 1024 | 1365 |

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

# Use DALL-E instead of mflux-serve
export IMAGE_BACKEND=openai
export OPENAI_API_KEY=sk-...
```

### Use explicit config in code

```python
from kg_utils.synthesis import TextSynthesizer, ImageSynthesizer
from kg_utils.synthesis._config import TextBackend, TextConfig, ImageBackend, ImageConfig

# Explicit Ollama config
text = TextSynthesizer(TextConfig(
    backend=TextBackend.OLLAMA,
    model="llama3:8b",
))

# Explicit DALL-E config
image = ImageSynthesizer(ImageConfig(
    backend=ImageBackend.OPENAI,
    api_key="sk-...",  # pragma: allowlist secret
))
```

### RAG pipeline integration

```python
from kg_utils.synthesis import text_synthesizer_from_env, image_synthesizer_from_env

text  = text_synthesizer_from_env()
image = image_synthesizer_from_env()

# Corpus hits from KGRAG / DocKG
hits = kgrag.query("great fire of London", k=8).hits

# Synthesise a grounded answer
answer = text.synthesize_rag(
    "What does Pepys say about the Great Fire?",
    [{"content": h.text, "author": h.author, "title": h.title} for h in hits],
    max_k=8,
)

# Rewrite the top hit into an image prompt, then generate
prompt, _ = text.rewrite_for_image(hits[0].text)
b64_png   = image.generate_b64(prompt, aspect_ratio="16:9", steps=6)
```

### Per-instance model cache (mflux-local)

`ImageSynthesizer` caches the loaded Flux2Klein model on the instance.  Create a single
synthesizer at startup and reuse it across requests to avoid redundant model loads:

```python
# At startup
_image = image_synthesizer_from_env()

# Per request — model already in memory after the first call
b64 = _image.generate_b64(prompt)
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
