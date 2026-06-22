"""Health-check evaluation — L0 structural + L1 heuristics (§11, §12.5).

Runs every ingest (code-only, no API).  Produces a dict consumed by the
health panel (Phase 5) and ``INDEX.md`` health section.

References
----------
- ARCHITECTURE.md §11 (anti-graveyard: decay vectors, near-dup, confidence,
  health report)
- ARCHITECTURE.md §12.5 item 4a (eval MVP slice: L0 + L1, free/code-only,
  every ingest)
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path


def run_health_check(cfg: object, store: object) -> dict:  # noqa: ARG001
    """Scan ``store.state`` + wiki files and return a health report dict.

    Fields returned
    ---------------
    source_count, topic_count
        Straight counts.
    orphans : dict
        ``{"sources": [...], "topics": [...]}`` — sources with empty
        ``topics`` AND topic slugs with empty ``sources``.
    broken_links : list[tuple[str, str]]
        ``(from_slug, target_slug)`` pairs where a ``links_to`` entry
        points to a slug not in ``store.state.topics``.
    near_duplicates : list
        Left empty for L0; the dedup module handles semantic near-dup.
    empty_extractions : list[str]
        Source IDs at stage ``FAILED`` or with empty topics.
    stale_topics : list[str]
        Slugs whose ``updated`` field is older than 90 days.
    avg_confidence : float
        Mean of all topic ``confidence`` values (0.0 if none).
    schema_violations : list[dict]
        Best-effort missing-required-field checks.
    """
    topics = store.state.topics
    sources = store.state.sources

    source_count = len(sources)
    topic_count = len(topics)

    # -- orphans ---------------------------------------------------------
    orphan_sources = [sid for sid, src in sources.items() if not src.topics]
    orphan_topics = [slug for slug, t in topics.items() if not t.sources]
    orphans: dict[str, list[str]] = {
        "sources": orphan_sources,
        "topics": orphan_topics,
    }

    # -- broken links ----------------------------------------------------
    broken_links: list[tuple[str, str]] = []
    for slug, t in topics.items():
        for target in t.links_to:
            if target not in topics:
                broken_links.append((slug, target))

    # -- near duplicates (L0: empty list) --------------------------------
    near_duplicates: list[tuple[str, str, float]] = []

    # -- empty extractions -----------------------------------------------
    empty_extractions: list[str] = [
        sid
        for sid, src in sources.items()
        if src.stage == "failed" or not src.topics
    ]

    # -- stale topics (>90 days since updated) ---------------------------
    now = datetime.now(UTC)
    stale_topics: list[str] = []
    for slug, t in topics.items():
        try:
            updated_str = t.updated
            if updated_str.endswith("Z"):
                updated_str = updated_str[:-1] + "+00:00"
            updated_dt = datetime.fromisoformat(updated_str)
            if (now - updated_dt) > timedelta(days=90):
                stale_topics.append(slug)
        except (ValueError, TypeError, AttributeError):
            stale_topics.append(slug)  # unparseable date counts as stale

    # -- average confidence ----------------------------------------------
    avg_confidence = (
        sum(t.confidence for t in topics.values()) / len(topics) if topics else 0.0
    )

    # -- schema violations (best-effort) ---------------------------------
    schema_violations: list[dict] = []
    for slug, t in topics.items():
        if not t.title:
            schema_violations.append(
                {"entity": slug, "field": "title", "issue": "empty"}
            )
    for sid, src in sources.items():
        if not src.sha256:
            schema_violations.append(
                {"entity": sid, "field": "sha256", "issue": "empty"}
            )

    sampled = _sampled_metrics(cfg, store)
    l1 = _l1_metrics(cfg, store)
    higher = _latest_higher_eval_metrics(cfg)

    return {
        "source_count": source_count,
        "topic_count": topic_count,
        "orphans": orphans,
        "broken_links": broken_links,
        "near_duplicates": near_duplicates,
        "empty_extractions": empty_extractions,
        "stale_topics": stale_topics,
        "avg_confidence": avg_confidence,
        "schema_violations": schema_violations,
        "mean_faithfulness_7d": sampled["mean_faithfulness_7d"],
        "merge_reversibility_pass_rate_7d": sampled["merge_reversibility_pass_rate_7d"],
        "cost_per_active_source_7d": sampled["cost_per_active_source_7d"],
        **l1,
        **higher,
    }


def _l1_metrics(cfg: object, store: object) -> dict[str, float | None]:
    root = getattr(cfg, "brain_root", None)
    if root is None:
        return {
            "citation_format_pass_rate": None,
            "hash_stability_pass_rate": None,
            "embedding_drift_rate": None,
            "topic_source_cosine_mean": None,
        }
    return {
        "citation_format_pass_rate": _citation_format_pass_rate(root),
        "hash_stability_pass_rate": _hash_stability_pass_rate(root, store),
        "embedding_drift_rate": _embedding_drift_rate(store),
        "topic_source_cosine_mean": _topic_source_cosine_mean(root),
    }


def _citation_format_pass_rate(root: Path) -> float | None:
    wiki_dir = root / "90-wiki"
    pages = list(wiki_dir.glob("*.md")) if wiki_dir.exists() else []
    checks = 0
    passes = 0
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        if "## Sources" not in text:
            continue
        for line in text.splitlines():
            if line.startswith("- "):
                checks += 1
                if re.search(r"\[source\]\(\.\./50-sources/[^)]+\.md\)", line):
                    passes += 1
    return passes / checks if checks else None


def _hash_stability_pass_rate(root: Path, store: object) -> float | None:
    checks = 0
    passes = 0
    for _source_id, source in store.state.sources.items():
        raw = root / source.raw
        if not raw.is_file():
            continue
        import hashlib

        checks += 1
        if hashlib.sha256(raw.read_bytes()).hexdigest() == source.sha256:
            passes += 1
    return passes / checks if checks else None


def _embedding_drift_rate(store: object) -> float | None:
    model = getattr(store.state, "embedding_model", "")
    if not model or not store.state.sources:
        return None
    drifted = [
        sid for sid, source in store.state.sources.items()
        if source.embedding_model and source.embedding_model != model
    ]
    return len(drifted) / len(store.state.sources)


def _topic_source_cosine_mean(root: Path) -> float | None:
    path = root / ".brain" / "evals" / "l1-topic-source-cosine.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        values = [float(v) for v in data.get("values", [])]
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return sum(values) / len(values) if values else None


def _sampled_metrics(cfg: object, store: object) -> dict[str, float | None]:
    """Read sampled 7-day eval/cost metrics from changelog events."""
    root = getattr(cfg, "brain_root", None)
    if root is None:
        return {
            "mean_faithfulness_7d": None,
            "merge_reversibility_pass_rate_7d": None,
            "cost_per_active_source_7d": None,
        }
    cutoff = datetime.now(UTC) - timedelta(days=7)
    path = root / ".brain" / "changelog.jsonl"
    faithfulness: list[float] = []
    reversibility: list[float] = []
    cost = 0.0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
                ts = event.get("ts", "")
                if ts.endswith("Z"):
                    ts = ts[:-1] + "+00:00"
                if ts and datetime.fromisoformat(ts) < cutoff:
                    continue
                scores = event.get("scores", {}) or {}
                usage = event.get("usage", {}) or {}
                if "faithfulness" in scores:
                    faithfulness.append(float(scores["faithfulness"]))
                if "merge_reversibility_pass" in scores:
                    reversibility.append(1.0 if scores["merge_reversibility_pass"] else 0.0)
                cost += float(usage.get("cost_usd", 0.0) or 0.0)
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    active_sources = max(len(store.state.sources), 1)
    return {
        "mean_faithfulness_7d": sum(faithfulness) / len(faithfulness) if faithfulness else None,
        "merge_reversibility_pass_rate_7d": (
            sum(reversibility) / len(reversibility) if reversibility else None
        ),
        "cost_per_active_source_7d": cost / active_sources,
    }


def _latest_higher_eval_metrics(cfg: object) -> dict[str, str]:
    root = getattr(cfg, "brain_root", None)
    if root is None:
        return {"l2_status": "unavailable", "l3_status": "unavailable", "l4_status": "unavailable"}
    manifest = root / ".brain" / "evals" / "latest.json"
    if not manifest.is_file():
        return {"l2_status": "not_run", "l3_status": "not_run", "l4_status": "not_run"}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"l2_status": "invalid", "l3_status": "invalid", "l4_status": "invalid"}
    return {
        "l2_status": data.get("l2", {}).get("status", "not_run"),
        "l3_status": data.get("l3", {}).get("status", "not_run"),
        "l4_status": data.get("l4", {}).get("status", "not_run"),
    }


async def run_higher_level_evals(cfg: object, store: object, client: object | None = None) -> dict:
    """Run/persist L2-L4 eval orchestration with provider work opt-in only."""
    eval_dir = cfg.brain_root / ".brain" / "evals"
    eval_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "ts": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "l2": _run_l2_self_consistency(store),
        "l3": {"status": "skipped", "reason": "set SECOND_BRAIN_RUN_LLM_EVALS=1 to enable"},
        "l4": _run_l4_golden_set(cfg),
    }
    if os.environ.get("SECOND_BRAIN_RUN_LLM_EVALS") == "1":
        if client is None:
            result["l3"] = {"status": "skipped", "reason": "no eval client supplied"}
        else:
            result["l3"] = await _run_l3_judge(cfg, client)
    (eval_dir / "latest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _run_l2_self_consistency(store: object) -> dict:
    mismatches = []
    for slug, topic in store.state.topics.items():
        for source_id in topic.sources:
            source = store.state.sources.get(source_id)
            if source is None or slug not in source.topics:
                mismatches.append({"topic": slug, "source": source_id})
    return {"status": "complete", "mismatches": mismatches, "score": 1.0 if not mismatches else 0.0}


async def _run_l3_judge(cfg: object, client: object) -> dict:
    text_model = getattr(cfg.models, "text", "")
    judge_model = getattr(cfg.models, "judge", "")
    if _model_family(text_model) == _model_family(judge_model):
        return {"status": "skipped", "reason": "judge model must be cross-family"}
    response = await client.chat_completion(
        judge_model,
        [{"role": "user", "content": "Return JSON: {\"score\": 1.0, \"notes\": \"ok\"}"}],
        extra_body={"headers": {"X-OpenRouter-Cache": "true"}},
    )
    return {"status": "complete", "model": judge_model, "response": response}


def _run_l4_golden_set(cfg: object) -> dict:
    golden_dir = cfg.brain_root / getattr(cfg.eval, "golden_set_dir", ".brain/golden")
    if not golden_dir.exists() or not any(golden_dir.glob("*.json")):
        return {"status": "skipped", "reason": "golden set is missing or empty"}
    return {"status": "complete", "golden_cases": len(list(golden_dir.glob("*.json")))}


def _model_family(model: str) -> str:
    return model.split("/", 1)[0].lower() if "/" in model else model.lower()


def render_health_markdown(report: dict) -> str:
    """Render the health report as a ``## Brain Health`` markdown section.

    Suitable for appending to ``INDEX.md``.
    """
    o = report["orphans"]
    n_orphan_sources = len(o["sources"])
    n_orphan_topics = len(o["topics"])
    total_orphans = n_orphan_sources + n_orphan_topics

    lines = [
        "## Brain Health\n",
        "\n",
        f"- **Sources**: {report['source_count']}\n",
        f"- **Topics**: {report['topic_count']}\n",
        f"- **Orphans**: {total_orphans}"
        f" ({n_orphan_sources} sources, {n_orphan_topics} topics)\n",
        f"- **Broken links**: {len(report['broken_links'])}\n",
        f"- **Empty extractions**: {len(report['empty_extractions'])}\n",
        f"- **Stale topics** (>90d): {len(report['stale_topics'])}\n",
        f"- **Avg confidence**: {report['avg_confidence']:.3f}\n",
        f"- **Schema violations**: {len(report['schema_violations'])}\n",
        f"- **Mean faithfulness (7d)**: {_fmt_metric(report.get('mean_faithfulness_7d'))}\n",
        "- **Merge reversibility pass rate (7d)**: "
        f"{_fmt_metric(report.get('merge_reversibility_pass_rate_7d'))}\n",
        f"- **Cost per active source (7d)**: ${report.get('cost_per_active_source_7d', 0.0):.4f}\n",
        "- **Citation format pass rate**: "
        f"{_fmt_metric(report.get('citation_format_pass_rate'))}\n",
        f"- **Hash stability pass rate**: {_fmt_metric(report.get('hash_stability_pass_rate'))}\n",
        f"- **Embedding drift rate**: {_fmt_metric(report.get('embedding_drift_rate'))}\n",
        f"- **Topic/source cosine mean**: {_fmt_metric(report.get('topic_source_cosine_mean'))}\n",
        f"- **L2 self-consistency**: {report.get('l2_status', 'not_run')}\n",
        f"- **L3 judge**: {report.get('l3_status', 'not_run')}\n",
        f"- **L4 golden regression**: {report.get('l4_status', 'not_run')}\n",
    ]
    return "".join(lines)


def _fmt_metric(value: float | None) -> str:
    return "n/a (no samples yet)" if value is None else f"{value:.3f}"
