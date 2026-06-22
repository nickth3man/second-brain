# Second Brain

A self-improving personal knowledge base inspired by Andrej Karpathy's setup:
dump information unstructured into the inbox; let the AI act as the librarian
that reads, links, and summarizes it into a living wiki. Value compounds over
time.

**Status: MVP functional with scale hardening implemented** — ingestion,
multimodal parsing, embeddings/retrieval, compaction, web UI, chat, long-source
map-reduce/RAPTOR extraction, startup vector reconcile, shadow embedding evals,
and health/eval surfacing are in place. Live LLM judge/golden evals remain
opt-in so normal tests stay offline.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e .

# Create your config from the tracked template, then add your OpenRouter key:
cp config.example.toml config.toml   # Windows: copy config.example.toml config.toml
brain init
```

The `brain init` command will walk you through API key setup and create the
required folder structure. It also checks OpenRouter's request-level ZDR preview
endpoint when possible, while clearly leaving account-level privacy toggles as a
manual confirmation item when no API metadata proves them.

## Supported ingestion

Second Brain parses text/code/structured files, PDFs/images, web/EPUB, audio,
video, modern Office documents, `.xlsx` spreadsheets, and `.pptx`
presentations. Legacy binary `.xls` and `.ppt` files are rejected with explicit
unsupported-format messages; save them as `.xlsx` or `.pptx` for safe parsing.

Video ingestion uses audio/STT as the primary signal and can optionally add a
bounded keyframe vision pass controlled by `[ingestion]` settings.

Long source extraction uses direct extraction under 16K tokens, explicit
map-reduce with a model reduce pass from 16K to 200K tokens, and RAPTOR-style
hierarchical summarization above 200K tokens.

## Configuration

`config.toml` holds your models and parameters — it is **gitignored** because it
contains your API key. The tracked [`config.example.toml`](config.example.toml)
is the template; copy it to `config.toml` and edit. See
[`ARCHITECTURE.md`](ARCHITECTURE.md) §9 for every slot.

API keys are resolved via Windows Credential Manager (keyring) → environment
variable → config file (§12.7). `brain init` writes the key to the keyring.

## Source of truth

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative design document.
All implementation decisions are recorded there.
