# Second Brain

A self-improving personal knowledge base inspired by Andrej Karpathy's setup:
dump information unstructured into the inbox; let the AI act as the librarian
that reads, links, and summarizes it into a living wiki. Value compounds over
time.

**Status: Phases 0–1 complete** — scaffold + the text-only ingestion loop
(watcher → extract → wiki → INDEX). Phase 2 (embeddings & retrieval) in progress.

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
required folder structure.

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
