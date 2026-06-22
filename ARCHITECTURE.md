# Second Brain — Architecture

> **Living build spec.** This document is the single source of truth for what gets
> built. The orchestrator keeps it up to date as design decisions are made.
> **Edits made by the user to this file are authoritative architectural decisions**
> and override prior conversation.

**Status:** MVP phases 0–6 implemented; architecture hardening and compliance pass in progress.
**Last updated:** 2026-06-22

---

## Table of Contents

1. [Vision & Principles](#1-vision--principles)
2. [Tech Stack](#2-tech-stack)
3. [Directory Structure](#3-directory-structure)
4. [Data Formats](#4-data-formats)
5. [The Pipeline (per-file)](#5-the-pipeline-per-file)
6. [File-Type Handling](#6-file-type-handling)
7. [Linking Model](#7-linking-model)
8. [Wiki Cadence & Compaction](#8-wiki-cadence--compaction)
9. [Configuration](#9-configuration)
10. [Query / Chat Interface](#10-query--chat-interface)
11. [Anti-Graveyard](#11-anti-graveyard)
12. [Implementation & Reliability](#12-implementation--reliability)
13. [Pending Design](#13-pending-design)
14. [Open Assumptions](#14-open-assumptions)

---

## 1. Vision & Principles

A self-improving personal knowledge base inspired by Andrej Karpathy's setup:
dump information unstructured; let the AI act as the librarian that reads,
links, and summarizes it into a living wiki. Value compounds over time.

**Core principles (non-negotiable):**

- **You are never the librarian.** The only thing the user does is drop files into
  `00-inbox/`. All organization, linking, and summarization is done by the AI.
- **Immutable originals.** `00-inbox/` is never mutated by anyone. Everything
  downstream is derived and therefore rebuildable.
- **Rebuildable wiki.** Because `50-sources/` is derived from `00-inbox/`, the
  entire `90-wiki/` + `INDEX.md` + `.brain/` can be wiped and regenerated from
  sources. This is the safety valve for full autonomy — bad topic structure can
  always be reset without data loss.
- **Fully autonomous.** No review queues, no confirmation prompts. The AI decides.
  (Escape hatch = rebuild from scratch, see above.)
- **Compounding.** A scheduled compaction pass continuously re-ranks, merges, and
  refreshes the wiki so day-100 > day-1.

---

## 2. Tech Stack

| Layer            | Choice                                              | Notes                                                                  |
| ---------------- | --------------------------------------------------- | ---------------------------------------------------------------------- |
| Language         | **Python** (3.12+)                                  | Locked by user decision; implemented in `pyproject.toml` |
| LLM backend      | **OpenRouter** (single API for all model calls)     | text, vision, embeddings, STT all via OpenRouter                       |
| PDF → image      | **PyMuPDF** (`pip install pymupdf`)                  | AGPL-3.0; acceptable because this project is open source               |
| File watcher     | `watchdog`                                          | triggers the daemon on `00-inbox/` events                              |
| Config           | TOML (`config.toml`)                                | all models + params swappable for testing                              |
| Vector store     | Local — embeddings cached in `.brain/`              | via OpenRouter embeddings endpoint (no local model needed)             |

**OpenRouter capabilities confirmed (2026-06-19):**
- `/embeddings` endpoint exists (OpenAI-compatible). Default: `openai/text-embedding-3-small`.
- STT via `openai/whisper-1`, or audio-as-input to a multimodal chat model.
- Vision providers cap **8 images per request** → pages sent one-per-request, sequentially.

---

## 3. Directory Structure

```
second-brain/
├── 00-inbox/          raw, immutable — the user's ONLY write target
├── 50-sources/        AI-normalized markdown (one file per inbox item)
├── 90-wiki/           living topic wiki (one page per topic)
├── INDEX.md           human-readable master map (the "one file")
├── config.toml        models + ingestion params (swappable)
└── .brain/            app state (machine sidecar)
    ├── state.json     topic graph, source registry, ingestion log
    └── changelog.jsonl  append-only audit log of compaction changes
```

| Path             | Written by     | Purpose                                                        |
| ---------------- | -------------- | -------------------------------------------------------------- |
| `00-inbox/`      | User only      | Raw dumps of any type. Never mutated.                          |
| `50-sources/`    | AI (ingestion) | Clean markdown extracted from each inbox item.                 |
| `90-wiki/`       | AI (ingestion + compaction) | Topic pages that grow as related sources accumulate. |
| `INDEX.md`       | AI             | Front door: recent changes, topic list, counts.                |
| `.brain/state.json` | AI          | Topic graph + source registry the app queries.                 |
| `.brain/changelog.jsonl` | AI (compaction) | Append-only audit of every compaction change.        |

---

## 4. Data Formats

### 4.1 Source file — `50-sources/<YYYY-MM-DD>-<slug>.md`

```markdown
---
source: 00-inbox/2026-06-15-rag-article.pdf      # relative path to raw original
type: pdf                                         # routed type
ingested: 2026-06-15T14:23:01Z                    # ISO 8601 timestamp
sha256: 3f2a9b...                                 # content hash (dedup)
tokens: 4218                                      # approx token count
topics: [rag-and-vector-search]                   # slugs this source feeds
---

# <extracted or original title>

<full normalized body — markdown>

## Summary
<2–3 sentence TL;DR of what this source says>
```

### 4.2 Wiki topic page — `90-wiki/<slug>.md`

```markdown
---
title: RAG & Vector Search                       # human name (AI-proposed); drives <h1>, breadcrumb, link display
slug: rag-and-vector-search                      # app-derived (see §10 slug rule)
type: concept                                    # drives the infobox renderer (see §4.6)
tags: [llm, retrieval, production]               # lightweight, AI-emitted (Track 8-1b)
aliases: [RAG, Retrieval-Augmented Generation]   # synonym redirects (Track 8-4a)
created: 2026-06-15
updated: 2026-06-19
source_count: 2
confidence: 0.82                                 # aggregate extraction confidence (Track 7-3a)
related: [llm-fundamentals, embedding-models]    # hand-curated cross-topic slugs
---

# RAG & Vector Search

<!-- Infobox: server-rendered from type + fields (§4.6). Not authored inline. -->

## Synthesis
<AI-rewritten each time a source is added — merged treatment of all linked
sources: definitions, how it works, tradeoffs, when to use.>

## Sources
- **[2026-06-19]** Vector Databases Explained (video)
  → [source](../50-sources/2026-06-19-vector-dbs.md) · [raw](../00-inbox/2026-06-19-vector-dbs.mp4)
  > 1–2 sentence summary of what THIS source contributed.

- **[2026-06-15]** Retrieval-Augmented Generation: A Practical Guide
  → [source](../50-sources/2026-06-15-rag-article.md) · [raw](../00-inbox/2026-06-15-rag-article.pdf)
  > 1–2 sentence summary of what THIS source contributed.

## Open questions
- <unresolved questions surfaced across sources>

## Related
- [[llm-fundamentals]]
- [[embedding-models]]

## See also
<!-- COMPUTED by the server from the wikilink graph ("what links here").
     Not authored. Surfaces pages whose ## Related or body links point here. -->

## Trivia           <!-- optional; render only if non-empty (Track 8-5a) -->
## Changes          <!-- optional; render only if non-empty (Track 8-5a) -->
```

**Field renames from the earlier draft:** `topic` → `title`, `last_updated` → `updated`
(aligned to the wiki-link spec in §10). A page's H1 always equals `title`.

**Sections:**
- **Mandatory** (authored): `## Synthesis`, `## Sources`, `## Open questions`, `## Related` (Track 2).
- **Computed** (server-generated, never authored): infobox (§4.6), `## See also`
  (Track 8-2a), breadcrumbs at top (Track 8-5a), tags footer.
- **Optional** (rendered only when non-empty): `## Trivia`, `## Changes` (Track 8-5a).

**Link styles:**
- Source links → relative markdown paths (`../50-sources/...`, `../00-inbox/...`)
- Cross-topic links → `[[wikilinks]]` per §10 (plain / piped / section / red-link)

### 4.3 `INDEX.md` (the front door)

```markdown
# Second Brain

**42 sources · 18 topics · last updated 2026-06-19**

## Recent
- 2026-06-19 → Vector Databases Explained (video)  [→ rag-and-vector-search]
- 2026-06-15 → Retrieval-Augmented Generation       [→ rag-and-vector-search]

## Topics
- [RAG & Vector Search](90-wiki/rag-and-vector-search.md) — 2 sources
- [LLM Fundamentals](90-wiki/llm-fundamentals.md) — 5 sources
```

### 4.4 `.brain/state.json`

```json
{
  "topics": {
    "rag-and-vector-search": {
      "title": "RAG & Vector Search",
      "type": "concept",
      "tags": ["llm", "retrieval", "production"],
      "aliases": ["RAG", "Retrieval-Augmented Generation"],
      "sources": ["2026-06-15-rag-article", "2026-06-19-vector-dbs"],
      "links_to": ["llm-fundamentals", "embedding-models"],
      "linked_from": [],
      "confidence": 0.82,
      "created": "2026-06-15",
      "updated": "2026-06-19"
    }
  },
  "sources": {
    "2026-06-15-rag-article": {
      "sha256": "3f2a9b...",
      "topics": ["rag-and-vector-search"],
      "raw": "00-inbox/2026-06-15-rag-article.pdf",
      "embedding_model": "openai/text-embedding-3-small"
    }
  }
}
```

### 4.5 `.brain/changelog.jsonl` (compaction audit)

One JSON object per line, append-only:

```json
{"ts":"2026-06-20T03:00:00Z","action":"merge","from":"vector-search","into":"rag-and-vector-search","similarity":0.88}
{"ts":"2026-06-20T03:00:01Z","action":"rewrite_synthesis","topic":"rag-and-vector-search","reason":"post-merge"}
```

### 4.6 Typed infobox schemas (Track 8-3c)

Each page's `type:` selects an infobox renderer. The AI assigns `type` during
extraction (defaults to `concept`). Schemas are **extensible** — add new types as
patterns emerge. The infobox is **server-rendered** from front-matter fields; it is
never authored inline in the markdown.

**Starter types:**

| `type`    | Infobox fields                                                              | Used for                          |
| --------- | --------------------------------------------------------------------------- | --------------------------------- |
| `concept` | Key idea (1-line), Source count, First seen, Updated, Confidence, Top sources | default — emergent topic clusters |
| `person`  | Role, Affiliation, Aliases, First mentioned, Sources                        | authors, contacts, figures        |
| `work`    | Author, Kind (book/article/paper/video), Year, Link, TL;DR, Sources         | things you consume                |
| `project` | Status, Started, Stack, Related topics, Sources                             | your projects                     |
| `tool`    | Category, Vendor/URL, License, First used, Sources                          | software, libraries, services     |
| `place`   | Location, Aliases, First mentioned, Sources                                 | locations                         |
| `event`   | Date, Aliases, Sources                                                      | dated occurrences                 |
| `note`    | (no infobox)                                                                | fleeting / raw thoughts           |

> Field values that reference other topics use `[[wikilinks]]`, so the infobox is
> itself part of the link graph (e.g. a `work` page's `Author: [[Andrej Karpathy]]`
> links into the graph and contributes to that person's computed `## See also`).

---

## 5. The Pipeline (per-file)

Runs **synchronously, one file at a time**, triggered by the file-watcher daemon.
Sequential by design (Track 2) — no batching, no parallelism.

```
TRIGGER ── new file detected in 00-inbox/ (watchdog)
   │
   ▼
[1] ROUTE by extension (see config.toml [types])
   ├── text/markdown   → passthrough
   ├── pdf             → PyMuPDF render pages → vision OCR (one page per request)
   ├── image           → vision model (describe + OCR)
   ├── code            → raw + AI summary
   ├── office/web/ebook→ parse → markdown
   ├── structured      → raw + AI summary
   ├── audio           → STT (whisper-1) → transcript
   └── video           → extract audio → STT (+ optional keyframes → vision)
   │
   ▼
[2] NORMALIZE → write 50-sources/<date>-<slug>.md  (front-matter + body + Summary)
   │
   ▼
[3] EXTRACT — text LLM reads normalized source, emits:
       • 3–7 candidate topics (human names)
       • 1-line TL;DR
       • key entities/concepts
   │
   ▼
[4] LINK — embed candidate topics; rerank vs existing 90-wiki/ page embeddings
           best similarity ≥ merge_threshold (0.70) → MERGE
           below                                   → SPAWN new topic
   │
   ▼
[5] UPDATE WIKI — write/merge 90-wiki/<slug>.md (rewrite-merge: AI regenerates
                  the whole Synthesis section incorporating the new source)
   │
   ▼
[6] UPDATE INDEX.md + cross-links + state.json   ← DEBOUNCED (see §8)
   │
   ▼
[7] (scheduled, not per-file) COMPACTION PASS    ← see §8
```

### Librarian prompt (skeleton — finalized in Track 6/7)

```
You are the librarian for a personal second brain.

INPUT:
- A normalized source markdown from 50-sources/.
- Existing topic pages in 90-wiki/ (titles + 1-line summaries).
- Current INDEX.md.

JOB:
1. Summarize the source in ≤2 sentences.
2. Extract 3–7 topics this source belongs to.
3. For each topic: MATCH to an existing wiki page if similarity high, else PROPOSE new.
4. Output the updated section to merge into each matched/new page.
5. Return strict JSON: { tldr, topics: [{ name, action: match|new, target_slug,
   confidence, merged_section }] }

CONSTRAINTS:
- Never invent facts not in the source.
- Quote sparingly; prefer compression.
- If unsure about a link, confidence < 0.6 and skip auto-merge.
```

---

## 6. File-Type Handling

### Ingestion matrix

| Type         | Extensions                                | Strategy                                          | Model stage  |
| ------------ | ----------------------------------------- | ------------------------------------------------- | ------------ |
| Plain text   | `.md` `.txt` `.markdown` `.rst`            | Passthrough (+ optional light cleanup)            | —            |
| PDF          | `.pdf`                                     | PyMuPDF render → vision OCR (one page per request)| Vision       |
| Images       | `.png` `.jpg` `.jpeg` `.webp` `.gif` ...   | Vision: describe + OCR                            | Vision       |
| Office docs  | `.docx` `.odt` `.rtf` `.pptx` ...          | Parse → markdown (mammoth / python-pptx)          | —            |
| Spreadsheets | `.xlsx` `.csv`                             | Sheets → markdown tables (visible cached values)  | —            |
| Web          | `.html` `.htm`                             | Readability extract → markdown                    | —            |
| Ebooks       | `.epub`                                    | Unzip XHTML → markdown                            | —            |
| Code         | `.py` `.js` `.ts` `.go` `.rs` ...          | Raw code + AI summary                             | Text LLM     |
| Structured   | `.json` `.yaml` `.toml` `.csv`             | Raw + AI summary                                  | Text LLM     |
| Audio        | `.mp3` `.m4a` `.wav` `.aac` `.ogg` `.flac`  | STT → transcript                                  | STT          |
| Video        | `.mp4` `.mov` `.mkv` `.webm` `.avi`        | Extract audio → STT (+ optional keyframes→vision) | STT + Vision |

### Model assignment (all via OpenRouter, all swappable in config.toml)

| Stage                                  | Default                                  |
| -------------------------------------- | ---------------------------------------- |
| Extract / summarize / link / synthesize| `anthropic/claude-3.5-sonnet`            |
| Vision (PDF OCR, images, keyframes)    | `openai/gpt-4o`                          |
| Embeddings (similarity/rerank)         | `openai/text-embedding-3-small`          |
| STT (audio/video)                      | `openai/whisper-1`                       |

### PDF rendering details (PyMuPDF)

- Renderer: `pymupdf` — `pip install pymupdf`, zero system deps, Windows-native wheels.
- DPI: `200` (sweet spot; vision models downscale to ~2048px edge anyway).
- Format: PNG (lossless, best OCR). Strip alpha (`alpha=False`, ~30% smaller).
- **One page per request** — OpenRouter vision caps 8 images/request; sequential
  sending also matches the synchronous pipeline.
- Prompt includes page number ("page 3 of 12") for layout faithfulness.
- License: **AGPL-3.0** — accepted for this open-source project. If a future
  closed-source binary/SaaS distribution is created, swap to `pypdfium2`
  (Apache-2.0) first.

### Office, Spreadsheet, and Presentation details

- `.xlsx` is parsed with `openpyxl` in `data_only=True` mode, so formulas are
  not executed; only cached visible values already stored in the workbook are
  emitted. Each sheet becomes a markdown heading plus a table, and empty sheets
  are represented explicitly.
- `.pptx` is parsed with `python-pptx`; slide text is ordered by top/left
  position where practical, slide numbers/headings are included, and speaker
  notes are emitted when present.
- Legacy binary `.xls` and `.ppt` are intentionally unsupported without an
  optional legacy converter; ingestion emits a clear unsupported-format error
  instead of a fake deferred note.

---

## 7. Linking Model

**Topics are the only structure.** No separate tag layer for MVP. (Track 4.)

```
NEW SOURCE
   │
   ▼
[EXTRACT]  text LLM → 3–7 candidate topics (human name + confidence)
   │
   ▼
[RERANK]   embed candidates; cosine-similarity vs existing 90-wiki/ embeddings
   │
   ▼
[DECIDE]   best similarity ≥ 0.70 → MERGE into existing page
           best similarity <  0.70 → SPAWN new page
   │
   ▼
[SLUG]     app slugs the AI-proposed name → rag-and-vector-search
           rule: lowercase, non-alphanumerics → hyphens, trimmed
   │
   ▼
[CROSSLINK] AI emits [[related-slugs]] → written to ## Related
            + edges logged in state.json (backlink graph)
```

- **Naming:** AI proposes the human name; app derives the slug. (Track 4-1a)
- **Merge threshold:** rerank similarity **≥ 0.70** merges; below spawns new. (Track 4-2a)
- **Tags:** lightweight AI-emitted wiki metadata. Track 8-1b supersedes the earlier
  Track 4-3a MVP simplification ("no tags").
- **Autonomy:** fully autonomous, no overrides. (Track 4-4b) Escape hatch = rebuild.
- **Backlinks:** `[[wikilinks]]` parsed and stored as graph edges in `state.json`
  (enables "what links here?" — nearly free since AI already emits links).

---

## 8. Wiki Cadence & Compaction

Two cadences: **immediate per-file** and **scheduled global**.

### Per-file (immediate)
- Stages [1]–[5] run synchronously on each detected file.
- Stage [6] (INDEX.md + cross-links + `state.json`) is **debounced** — flushes
  30s after the last file settles. (Track 5-1a) Feels instant on single drops,
  cheap on bulk.

### Scheduled compaction (the self-improvement engine)
- **Trigger:** daily OR every 25 new sources, whichever comes first. (Track 5-2a)
- **Aggressiveness:** conservative. (Track 5-3a)
  - Merge topics only when similarity **≥ 0.85**.
  - **Never delete** pages.
  - Rewrite stale `## Synthesis` sections.
  - Refresh `## Open questions` and `## Related`.
  - **Log every change** to `.brain/changelog.jsonl` (append-only, auditable).

---

## 9. Configuration

All models and parameters live in [`config.toml`](./config.toml). Key slots:

- `[openrouter]` — `base_url`, `api_key`
- `[models]` — `text`, `vision`, `embedding`, `stt` (each swappable, with commented alternatives + prices)
- `[ingestion]` — `merge_threshold` (0.70), `pdf_dpi` (200), `pdf_image_format` (png),
  `pdf_alpha` (false), `vision_max_images_per_request` (8), `vision_max_edge_px` (2048),
  `max_audio_minutes` (120)
- `[types]` — extension → stage routing

Swap any value to test alternatives without touching code.

---

## 10. Query / Chat Interface

The payoff: "when I ask for help, it's working from years of accumulated context
instead of a blank page." Delivered via a **web UI** (FastAPI + minimal frontend)
that both browses the wiki and hosts a chat panel.

### Query flow — agentic RAG (not a single LLM call)

```
brain ask "how do I chunk documents for RAG?"   (web UI chat)
   │
   ▼
[AGENT LOOP]  text LLM + tools decides how to investigate:
   • search_brain(query)  → embed query; cosine-sim over 90-wiki/ + 50-sources/;
                             return top-K passages
   • get_topic(slug)      → fetch a full topic page
   • get_sources(topic)   → list sources for a topic
   │
   ▼
[STREAM]  tokens streamed live; UI shows the agent trace:
            • THINKING   (reasoning tokens, where the model exposes them)
            • TOOL CALLS (each search_brain / get_topic + what it returned)
            • FINAL ANSWER
   │
   ▼
[ANSWER]  grounded synthesis + inline citations → [source] / [[topic]]
```

### Decisions (Track 6)
- **Interface — Web UI** (FastAPI + minimal frontend). (6-1c)
- **Answer shape — synthesis with citations.** Grounded prose + inline links to
  the sources/topics it drew from. (6-2a)
- **Streaming + visible thinking + tool calls.** Stream tokens live via
  OpenRouter streaming; show the agent's reasoning and every retrieval step so
  each citation is auditable/trustworthy. (6-3a)
- **Read-only.** Queries never write back to the brain. (6-4c) Rationale: if a
  Q&A needs saving to be useful, the brain isn't navigable enough — fix the
  brain, don't patch with chat logs.

### Wiki-link rendering *(spec from lib-2, grounded in OSRS / MediaWiki)*

**Source syntax** — the ONLY syntax for cross-topic links:
| Form     | Source                    | Renders as                                                |
| -------- | ------------------------- | --------------------------------------------------------- |
| Plain    | `[[Page Name]]`           | blue link, display = page name                            |
| Piped    | `[[Page Name\|display]]`  | blue link, custom display (e.g. `[[Anvil\|anvils]]`)        |
| Section  | `[[Page Name#section]]`   | blue link to anchor                                       |
| Missing  | `[[Unwritten Topic]]`     | **red link** → `/topic/<slug>?action=create`                |
| External | `[label](https://…)`      | external-link style — NOT for topics                       |

> **Rule:** never use `[text](url)` for cross-topic refs — only `[[wikilinks]]`.
> Only `[[…]]` lets the renderer detect missing pages (red links) and compute backlinks.

**Slug rule** (page name → URL):
```
slug = page_name.lower().replace(' ', '-').replace('_', '-').strip('-')
       # also strip parens & apostrophes
# "The Tourist Trap"            -> the-tourist-trap
# "Barrel (The Tourist Trap)"   -> barrel-the-tourist-trap
# "Members' NPCs"               -> members-npcs
```
URL pattern: `/topic/{slug}` (create: `/topic/{slug}?action=create`). Titles keep
original capitalization for display; the `title=""` attribute always shows the human
page name (mirrors OSRS).

**Rendering matrix (FastAPI side):**
| State    | Output                                                                                  | Class                            |
| -------- | --------------------------------------------------------------------------------------- | -------------------------------- |
| Exists   | `<a href="/topic/{slug}" title="{page}">{display}</a>`                                   | `wikilink` (blue)                |
| Missing  | `<a href="/topic/{slug}?action=create" title="{page} (page does not exist)">{display}</a>` | `wikilink wikilink--missing` (red, dotted underline) |
| Self     | `<span class="wikilink wikilink--self">{page}</span>` (bold, not clickable)               | `wikilink--self`                 |
| External | `<a target="_blank" rel="noopener" class="extlink">…</a>`                                | `extlink`                        |

**Worked example (plain + piped + red link):**
```markdown
He appears in [[The Tourist Trap]] and uses the [[Anvil|anvils]] to make
a [[Mythril bar]] prototype.   ← "Mythril bar" has no page -> red link
```
→ *The Tourist Trap* and *Anvil* render blue; *Mythril bar* renders red with a create-link.

> **lib-2 page-structure refinements adopted by Track 8:** YAML `tags:`, `type:`
> + auto infobox, `aliases:` redirects, computed `## See also` backlinks, and
> breadcrumbs are now part of §4.2/§4.6/§10. Track 8-1b supersedes the earlier
> Track 4-3a MVP simplification ("no tags").

---

## 11. Anti-Graveyard

Explicit defense against the transcript's failure mode: *"every second brain setup
ends the same way... graveyard in a few weeks."* Layered on two existing defenses
(§8 compaction pass; §7 full autonomy so you're never the librarian).

### Decay vectors → mechanism
| Vector         | Mechanism                                                  |
| -------------- | ---------------------------------------------------------- |
| Duplicates     | sha256 exact-dedup + embedding near-dup detection (below)  |
| Fragmentation  | compaction merges topics ≥ 0.85 similarity (§8)            |
| Staleness      | flagged when a topic untouched > stale threshold           |
| Orphans        | sources/topics with zero links flagged                     |
| Low confidence | every extraction tagged; low-conf flagged                  |

### Decisions (Track 7)
1. **Source dedup — exact + semantic.** (7-1a) Exact `sha256` match → skip ingest
   silently. Near-duplicate (embedding cosine ≥ 0.95) → ingest, cross-link to the
   existing source, badge as near-duplicate. Batch near-dup scans use a
   deterministic LSH pre-filter at 1000+ sources to avoid O(n²) full comparisons
   during normal scale operation; per-source ingest uses existing source
   centroids/cached embeddings.
2. **Surfacing — web UI health panel, passive.** (7-2a) Orphans, stale topics,
   low-confidence items, duplicates shown as non-blocking badges. Never a review
   queue (honors §7 full autonomy).
3. **Confidence scoring.** (7-3a) Every extraction carries a 0.0–1.0 score.
   Low-confidence items flagged visibly but still auto-accepted. Scores also
   surface as a trust signal on citations in §10 answers.
4. **Health report.** (7-4a) Compaction emits a summary (source/topic counts,
   orphans, duplicates, stale count, avg confidence) to BOTH the web UI dashboard
   AND a `## Brain Health` section appended to `INDEX.md` each run.

### Concrete thresholds *(adjustable)*
- Topic **stale** if `last_updated` > 90 days.
- Nothing is ever auto-deleted (consistent with Track 5-3a "never delete").

---

## 12. Implementation & Reliability

> Derived from a 7-domain research pass (lib-3…lib-9, 2026-06-19) + Five Whys
> root-cause analysis (Root A = "improving blindly"; Root B = "unvalidated
> load-bearing choices"). Consolidated decisions: **Block A accepted**; **Block B
> = 1a · 2a · 3a · 4a · 5no · 6a**.

### 12.1 Vector store — sqlite-vec  *(lib-3, decision 1a)*
- **sqlite-vec** — embedded, Windows-native wheels, single `.brain/embeddings.db`, MIT. Beats Chroma (CVE-2026-45829, CVSS 10, unpatched + messy migrations), Milvus Lite (immature single-process lock), FAISS (Windows pip pain), Qdrant (needs a server).
- Schema: `model_registry` (active model + dim, append-only) · `source_chunks_vec` (`vec0` float[1536] cosine) · `topic_centroids_vec` · `topic_members` · `source_chunks_fts` (FTS5 BM25 mirror) · `vec_tombstones`.
- **Hybrid retrieval** for `search_brain`: vector + FTS5 merged via **RRF (k=60)**. Topic-merge stays **vector-only** (semantic; no lexical overlap expected). int8 quantization at 10k+ scale (~35–45 MB).
- **Concurrency (1a):** the single-process SQLite lock means the **daemon owns ALL writes**; the FastAPI UI opens **no SQLite connection at all** and calls the daemon over **loopback HTTP** for `search_brain` / `reindex` / `compact`. `search_brain` lives in the daemon, not in FastAPI.

### 12.2 Structured output & extraction reliability  *(lib-4)*
- Mechanism: `response_format: json_schema strict` (Pydantic `model_json_schema()`) + OpenRouter **`response-healing`** plugin (free, 80%+ JSON-defect reduction) + `provider.require_parameters: true` + Pydantic **`extra="forbid"`** + `model_validate_json()`.
- **Schema stays static** — never bake the topic list into the schema as an enum (kills the grammar cache, hits param caps). Topic-matching stays in the embedding layer (§7).
- Retry: model-fallback × retry double loop; retry **only** 429/5xx/connection; 4xx → next model; refusal/max_tokens → dead-letter. Dead-letters → `.brain/deadletter/`; all-low-confidence → `.brain/quarantine/`. **Never** write a partial `LibrarianOutput` to the wiki.
- Long sources: <16K tokens direct extraction; 16K–200K map-reduce
  (approximately 800-token chunks, 100-token overlap) with an explicit
  schema-validated reduce pass; 200K+ RAPTOR-style hierarchical summarization
  builds deterministic chunk-summary trees with traceability back to original
  chunks. No truncation/deferred path may write partial output.

### 12.3 Daemon & ingestion robustness  *(lib-5)*
- **Stable-file gate:** `on_moved` is first-class (PyCharm/VSCode/Word emit it, not `on_modified`); ignore temp-file regex (`~$*`, `.goutputstream`, `.crdownload`, `.part`, `.swp`); stable = exists + size>0 + exclusive-open succeeds + size/mtime unchanged across two polls.
- **Per-source state machine** keyed on `sha256` (seen→hashing→normalized→extracted→linked→wiki_merged→indexed→done, + `failed`). `state.json` **IS** the WAL → idempotent re-runs.
- **Atomic writes:** same-dir `NamedTemporaryFile` + `flush` + `fsync` + `os.replace` (validate-before-replace). Win32-aware: `os.replace` not `os.rename`; may fall back to copy → **backups are mandatory**, not optional.
- **Per-page PDF checkpoint** via `.brain/cache/<sha>.progress.json` sidecar → a failed 200-page PDF resumes from page 151, not page 1.
- **OpenRouter failures:** tenacity + honor `Retry-After`; **402 (credit exhaustion) stops the daemon** + health-panel alert.
- **Startup-reconcile pass** = mandatory self-healing (state↔filesystem both
  directions plus vector/index drift repair). It detects and repairs missing,
  orphan, and stale source embeddings; FTS mirror drift; missing/stale topic
  centroids; topic membership drift; and tombstone inconsistencies while
  preserving the daemon single-writer boundary.

### 12.4 Web UI & agent stack  *(lib-6)*
- Stack: **FastAPI ≥0.135** (native `EventSourceResponse`) + Jinja2 + **HTMX 2** (`htmx-ext-sse`) + **mistune 3** (~35-line custom wikilink plugin) + **Pydantic AI** (`run_stream_events()`). **No SPA, no build step.**
- Wikilink render pipeline: load page index → split front-matter → strip computed sections → pre-compute breadcrumbs/infobox → mistune render (4 link states per §10) → Jinja assemble.
- SSE event schema: `thinking` / `tool_call` / `tool_result` / `answer_delta` / `done`. HTMX **append-fix** (~6 lines JS) so `answer_delta` appends rather than replaces.
- Agent loop: **Pydantic AI** (its event types map 1:1 to our SSE schema; handles OpenRouter tool-calling + message history). `search_brain` / `get_topic` as `@tool_plain`.
- **Reasoning visibility (2a):** default chat model must surface `reasoning_details` → **Anthropic / DeepSeek / Gemini — NOT OpenAI o-series** (o-series hides reasoning). Pass `reasoning_details` back on every tool-call turn (Pydantic AI does this).

### 12.5 Observability, evals & cost — eval MVP slice  *(lib-7, decision 4a)*
- **Layered eval harness — measured + surfaced in the health panel, NEVER gated** (honors autonomy):
  - **L0 structural invariants** (code, free, every ingest): orphans, broken links, dupes, empty extractions, schema violations.
  - **L1 heuristics** (code, free, every ingest): citation format, hash stability, embedding drift, topic↔source cosine.
  - **3 headline metrics (sampled):** `mean_faithfulness_7d` (RAGAS, per chat) · `merge_reversibility_pass_rate_7d` (per compaction) · `cost_per_active_source_7d`.
  - L2–L4 (self-consistency, full LLM-as-judge, golden-set regression) are
    implemented as persisted eval artifacts. L2 is offline/deterministic; L3
    LLM judge is opt-in via environment/config and enforces a cross-family judge;
    L4 consumes cached golden-set fixtures and skips honestly when the set is
    absent.
- **Judge model must be cross-family** (e.g. Claude generates, GPT judges) — same-family judges rate their own output ~10–15% high (the `judge` slot, decision 2a).
- **Tracing:** extend `.brain/changelog.jsonl` with `kind` (ingest|merge|chat|compact|eval|judge) + `trace_id`/`span_id`/`parent_span_id` + per-event `usage`/`scores`/`manifest`. No separate observability backend.
- **Cost/caching:** `session_id` per compaction/chat run (prompt-cache sticky routing); OpenRouter **response-cache** (`X-OpenRouter-Cache`) for golden runs (zero-cost after first); embedding cache keyed on `(model, sha256, text)`. Caps: per-call `max_tokens`, per-run counter, OpenRouter dashboard hard cap.

### 12.6 Data integrity, migration & rebuild  *(lib-8)*
- **Atomic writes + backups:** rolling 3-deep (`state.json`/`.bak`/`.bak-1`/`.bak-2`) + daily `.brain/snapshots/` + load-time pydantic recovery (primary→bak→bak-1→rebuild). Single-writer in-process lock (no file lock).
- **`schema_version`** in BOTH front-matter and `state.json` root; pydantic `model_validator(mode="before")` transparent in-load migration; historical migrations kept as tested code forever; **in-place for syntactic renames, rebuild for semantic changes.**
- **`brain rebuild`** command: `--from-sources` (fast, re-runs link, ~\$0.08/1k) · `--from-inbox` (deep, re-runs extract, ~\$2–5/1k) · `--dry-run`; implicit backup to `.brain/snapshots/pre-rebuild-<ts>/`; **atomic-delete-then-rebuild** (crash leaves a complete prior state); idempotent. Always preserves `00-inbox/`, `config.toml`, `changelog.jsonl`, git history.
- **Embedding swap = blue/green:** audit → cached shadow eval test-set →
  build `.brain/embeddings.new/` → atomic dir-swap + `state.json` update
  (model + dim + per-source `embedding_model`) → validate vs cached test set →
  **rollback if >10% worse**; never overwrite in place;
  `embedding_swap_in_progress` flag for crash-resume. Eval manifests are written
  under `.brain/evals/`.

### 12.7 Privacy & security — OpenRouter-only  *(lib-9, decision 5no)*
- **`provider.zdr: true` on EVERY request** (stronger than `data_collection: deny` — ZDR filters retention, not just training). `brain init` checks the documented OpenRouter request-level ZDR preview endpoint when available, but does not claim account-level dashboard toggles are verified unless OpenRouter exposes reliable account metadata; otherwise they remain manually confirmed/unconfirmed.
- **STT swap (2a):** `whisper-1` has **no ZDR endpoint** → default `stt = "openai/whisper-large-v3"` via `provider.only: ["groq","together"]` (ZDR). *(user may override in config.toml)*
- **No PII redaction** (destroys extraction quality); `00-inbox/sensitive/` path flag → strictest ZDR routing. **5no = no local-model fallback** — sensitive media routes to ZDR cloud, not on-box. Accepted threat model: ZDR permits RAM caching; first-party retention terms vary by provider.
- **API key:** Windows Credential Manager via `keyring` (DPAPI); `brain init` writes it; resolved keyring → env → config. *(user may paste directly in config as fallback)*
- **STT resumability (3a):** **ffmpeg + chunked STT** (resumable per-chunk; adds ffmpeg dep + minor boundary artifacts).
- **Licensing:** stay **PyMuPDF** (AGPL) for this open-source project; PDF render
  wrapped behind a `pdf.render()` interface so an AGPL→pypdfium2(Apache) swap is
  mechanical if a future closed-source/binary/SaaS distribution requires it.
- **Hardening:** FastAPI bound to **127.0.0.1 only**; sqlite-vec chosen over ChromaDB (CVE-2026-45829).

### 12.8 Versioning — git  *(decision 6a, amended for privacy policy)*
- Single repo at `second-brain/` root, `main` branch only.
- **Versioned (project tooling + spec):** `ARCHITECTURE.md`, source code under `src/`, `tests/`, `pyproject.toml`, `uv.lock`, `README.md`, and `config.example.toml`. These capture the brain's *behavior* and architecture.
- **Derived brain content is user-private and ignored:** `50-sources/`, `90-wiki/`, `INDEX.md`, `.brain/state.json`, `.brain/changelog.jsonl`. This keeps each user's knowledge base local/private and avoids binary/generated churn in git. Recovery of derived content is via the `brain rebuild` command (§12.6) rather than `git checkout`.
- **Commit cadence:** commit on the compaction pass, scoped to changed versioned project files only (spec, source, example config). No-op if only derived/ignored files changed.
- **Ignore:** `.brain/embeddings.db` (+ `-journal`/`-wal`/`-shm`), `.brain/cache/`, `.brain/snapshots/`, `.brain/deadletter/`, `.brain/quarantine/`, `.env`, `*.local.toml`, `.slim/deepwork/`, OS/editor cruft.
- **`config.toml` ignored** — contains the user's API key and model choices; use `config.example.toml` as the tracked template.
- **`00-inbox/` ignored** (6a) — binary originals bloat `.git` forever; back up separately (OneDrive/rsync/etc.).

---

## 13. Implementation Status

- **Implementation phases** — MVP phases 0–6 have code and tests in this repo:
  - **0. Scaffold** — implemented: config loader, OpenRouter client, `keyring`, folder init.
  - **1. Text-only ingestion loop** — implemented and tested.
  - **2. Linking & embeddings** — implemented with sqlite-vec, graph state, hybrid retrieval.
  - **3. Multimodal** — implemented: PDF/image/office/web/ebook/audio/video
    routing and parsers, including `.xlsx`, `.pptx`, and optional video
    keyframe vision.
  - **4. Compaction & health** — implemented with L0/L1 metrics, offline L2,
    opt-in L3 judge, L4 golden-set orchestration, scale-ready source near-dup
    prefiltering, and startup vector/index reconcile.
  - **5. Web UI** — implemented with FastAPI + HTMX + mistune rendering.
  - **6. Chat** — implemented with read-only streaming agent events.
- **Compliance posture** — §12 hardening items are verified continuously by
  code/tests; provider-backed evals remain opt-in to keep normal test runs
  deterministic and offline.

---

## 14. Open Assumptions

> No open assumptions remain as of 2026-06-22. Everything below is a locked decision.

| Item                        | Value                                  | Status                              |
| --------------------------- | -------------------------------------- | ----------------------------------- |
| Language                    | Python 3.12+                           | Decided; implemented in tooling |
| PDF renderer license        | PyMuPDF (AGPL), open-source project; wrapped behind `pdf.render()` iface | Decided; compatible with current open-source intent |
| Interface                   | Web UI (FastAPI + Jinja2 + HTMX + mistune + Pydantic AI) | Decided (6-1c, §12.4)          |
| Query write-back            | Read-only                              | Decided (Track 6-4c)                |
| Compaction trigger          | daily OR every 25 sources              | Decided (Track 5-2a), adjustable    |
| Debounce window             | 30 seconds                             | Decided (Track 5-1a), adjustable    |
| Stale threshold             | 90 days                                | Decided (Track 7), adjustable       |
| Slug rule                   | lowercase, hyphenated, alnum only; strip parens & apostrophes | Decided (§10 refined)            |
| Date format                 | ISO 8601                               | Decided (convention)                |
| Tags                        | lightweight, AI-emitted, browsable `/tags/<slug>` pages | Decided (Track 8-1b; revises 4-3a) |
| Infobox                     | typed, per-type schema (§4.6)          | Decided (Track 8-3c), extensible    |
| Computed backlinks (`## See also`) | server-generated from link graph | Decided (Track 8-2a)                |
| Aliases / breadcrumbs       | adopted                                | Decided (Track 8-4a / 8-5a)         |
| Vector store                | sqlite-vec (embedded, `.brain/embeddings.db`) | Decided (§12.1)               |
| Daemon owns DB writes       | UI read-only + loopback HTTP to daemon | Decided (Block B-1a)               |
| Text model                  | bumped `claude-3.5-sonnet` → `claude-sonnet-4.5` (structured outputs) | Decided (Block B-2a) |
| Models config               | `[models]` slots: text/vision/embedding/stt/**chat/judge** (user fills preferred + key) | Decided (Block B-2a) |
| STT                         | `whisper-large-v3` via Groq/Toaster (ZDR); ffmpeg-chunked for resumability | Decided (Block B-2a/3a) |
| Eval scope                  | MVP slice plus production-scale L2–L4 required before broad 100–2000+ source use | Decided (updated 2026-06-22) |
| Privacy                     | `provider.zdr:true` everywhere; no redaction; OpenRouter-only (no local fallback) | Decided (Block B-5no) |
| Versioning                  | git the brain; `00-inbox/` ignored     | Decided (Block B-6a)                |

---

*Decisions log — Tracks 1–8 + §12 reliability research locked (2026-06-19).
Edit this file directly to make or revise architectural decisions.*
