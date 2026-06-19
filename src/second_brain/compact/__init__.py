"""Compaction, health evaluation, and near-duplicate detection (§8, §11, §12.5)."""

from second_brain.compact.compaction import run_compaction  # noqa: F401
from second_brain.compact.dedup import find_near_duplicates  # noqa: F401
from second_brain.compact.eval import render_health_markdown, run_health_check  # noqa: F401
