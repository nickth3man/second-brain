# Second Brain

A self-improving personal knowledge base inspired by Andrej Karpathy's setup:
dump information unstructured into the inbox; let the AI act as the librarian
that reads, links, and summarizes it into a living wiki. Value compounds over
time.

**Status: Phase 0 scaffold** — config loader, OpenRouter client, keyring
integration, folder init. Ingestion daemon coming in Phase 1.

## Quick start

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -e .
brain init
```

The `brain init` command will walk you through API key setup and create the
required folder structure.

## Configuration

All models and parameters live in [`config.toml`](config.toml). Swap any
model, adjust ingestion settings, or change privacy behaviour without touching
code. See [`ARCHITECTURE.md`](ARCHITECTURE.md) §9 for details.

API keys are resolved via Windows Credential Manager (keyring) → environment
variable → config file (§12.7).

## Source of truth

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the authoritative design document.
All implementation decisions are recorded there.
