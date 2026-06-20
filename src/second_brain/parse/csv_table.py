"""CSV/JSON summarizer — produces compact markdown for tabular datasets.

Used in the normalize stage for ``structured``-type files (CSV, TSV, JSON).
No pandas/numpy dependency — stdlib ``csv`` and ``json`` only.
"""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path
from typing import Any

MAX_DISPLAY_ROWS = 5
MAX_DISPLAY_COLS = 10
MAX_CELL_CHARS = 60
DISPLAY_CAP = 100_000


# -- type inference -----------------------------------------------------------


def _infer_type(values: list[str]) -> str:
    """Infer the most specific type for a column from sample values."""
    types_found: set[str] = set()
    for v in values:
        v = v.strip()
        if not v:
            continue
        try:
            int(v)
            types_found.add("int")
            continue
        except ValueError:
            pass
        try:
            float(v)
            types_found.add("float")
            continue
        except ValueError:
            pass
        if _looks_like_date(v):
            types_found.add("date")
            continue
        if v.lower() in ("true", "false", "yes", "no", "t", "f", "y", "n", "0", "1"):
            types_found.add("bool")
            continue
        types_found.add("string")

    if not types_found:
        return "string"
    if types_found == {"int"}:
        return "int"
    if types_found == {"float"} or types_found == {"int", "float"}:
        return "float"
    if types_found == {"date"}:
        return "date"
    if types_found == {"bool"}:
        return "bool"
    return "mixed"


def _looks_like_date(v: str) -> bool:
    """Heuristic: YYYY-MM-DD or common date separators (no expensive parse)."""
    v = v.strip()
    parts = v.split("-")
    if len(parts) == 3 and len(parts[0]) == 4 and all(p.isdigit() for p in parts):
        return True
    parts = v.split("/")
    return bool(len(parts) == 3 and all(p.isdigit() for p in parts))


# -- CSV summarizer -----------------------------------------------------------


def _truncate_cell(val: str, max_chars: int = MAX_CELL_CHARS) -> str:
    """Truncate a cell value for display, appending ``...`` if needed."""
    val = val.replace("\n", "\\n").replace("\r", "")
    if len(val) > max_chars:
        return val[: max_chars - 3] + "..."
    return val


def summarize_csv(
    path: Path,
    *,
    max_sample_rows: int = MAX_DISPLAY_ROWS,
    max_cell_chars: int = MAX_CELL_CHARS,
) -> str:
    """Produce a compact markdown summary of a CSV/TSV file.

    Args:
        path: Path to the CSV/TSV file.
        max_sample_rows: Maximum data rows to include as a sample table.
        max_cell_chars: Maximum characters per cell in the output.

    Returns:
        A markdown string describing the file.
    """
    raw = path.read_bytes()
    # Detect delimiter from extension; default to comma.
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    try:
        header = next(reader)
    except StopIteration:
        return f"# {path.stem}\n\n_Empty file._\n"

    n_cols = len(header)
    # Reservoir sampling (seeded -> deterministic) so the sample represents
    # the WHOLE file, not just the first N rows. Without this, a
    # heterogeneous file (e.g. a multi-file data dictionary) yields a sample
    # from only its first section, biasing the topic title toward it.
    rng = random.Random(20250620)
    sampled: list[tuple[int, list[str]]] = []  # (row index, row)
    row_count = 0

    for row in reader:
        # Skip empty rows
        if not row or all(c.strip() == "" for c in row):
            continue
        i = row_count
        row_count += 1
        if len(sampled) < max_sample_rows:
            sampled.append((i, row))
        else:
            j = rng.randint(0, i)
            if j < max_sample_rows:
                sampled[j] = (i, row)

    sampled.sort(key=lambda t: t[0])
    rows = [r for _, r in sampled]

    # Cap display for huge files
    row_display = f"{DISPLAY_CAP:,}+" if row_count > DISPLAY_CAP else f"{row_count:,}"

    lines: list[str] = []
    lines.append(f"# {path.stem}")
    lines.append("")
    # Handle ragged rows
    ragged = any(len(r) != n_cols for r in rows)
    if ragged:
        lines.append(
            f"_{row_display} rows x {n_cols} columns (some rows have "
            f"inconsistent column counts — values shown best-effort)_"
        )
    else:
        lines.append(f"_{row_display} rows x {n_cols} columns_")
    lines.append("")

    # Column table: Column | Type | Sample value
    lines.append("| Column | Type | Sample value |")
    lines.append("|--------|------|--------------|")

    # Collect sample values per column (from sample rows)
    n_sample_cols = min(n_cols, MAX_DISPLAY_COLS)
    col_samples: list[list[str]] = [[] for _ in range(n_sample_cols)]
    for row in rows:
        for ci in range(min(len(row), n_sample_cols)):
            col_samples[ci].append(row[ci])

    truncated_cols = n_cols > MAX_DISPLAY_COLS
    for ci in range(n_sample_cols):
        col_name = header[ci] if ci < len(header) else f"col_{ci}"
        sample_vals = col_samples[ci] if ci < len(col_samples) else []
        col_type = _infer_type(sample_vals)
        sample = _truncate_cell(sample_vals[0], max_cell_chars) if sample_vals else ""
        lines.append(f"| {col_name} | {col_type} | {sample} |")

    if truncated_cols:
        remaining = n_cols - MAX_DISPLAY_COLS
        lines.append(f"| _… {remaining} more columns_ | | |")

    lines.append("")

    # Sample rows table
    if rows:
        lines.append("### Sample rows")
        lines.append("")
        # Header row
        hdr = [header[ci] if ci < len(header) else f"col_{ci}" for ci in range(n_sample_cols)]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("| " + " | ".join(["---"] * n_sample_cols) + " |")
        for row in rows[:max_sample_rows]:
            display_row = [
                _truncate_cell(row[ci] if ci < len(row) else "", max_cell_chars)
                for ci in range(n_sample_cols)
            ]
            lines.append("| " + " | ".join(display_row) + " |")
        if truncated_cols:
            lines.append("")
            lines.append(
                f"_Showing first {MAX_DISPLAY_COLS} of {n_cols} columns._"
            )
        if row_count > max_sample_rows:
            lines.append("")
            lines.append(
                f"_Showing {min(len(rows), max_sample_rows)} of {row_count} data rows._"
            )

    lines.append("")
    return "\n".join(lines)


# -- JSON summarizer ----------------------------------------------------------


def _json_shape(value: Any, indent: int = 0) -> str:
    """Return a compact type/shape description of a JSON value."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in value.items():
            items.append(f"{pad}  {k}: {_json_shape(v, indent + 1)}")
        return "{\n" + "\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        inner = _json_shape(value[0], indent)
        return f"[{inner}, ...] ({len(value)} items)"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return "string"


def summarize_json(path: Path) -> str:
    """Produce a compact markdown summary of a JSON file.

    Shows the top-level type, key count, and a compact tree for objects.
    """
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return f"# {path.stem}\n\n_Invalid JSON: {e}_\n"

    lines: list[str] = []
    lines.append(f"# {path.stem}")
    lines.append("")

    if isinstance(data, dict):
        n_keys = len(data)
        lines.append(f"JSON object with **{n_keys}** top-level keys:")
        lines.append("")
        lines.append("```")
        lines.append(_json_shape(data))
        lines.append("```")
        # Sample a few key-value pairs as a table
        sample_items = list(data.items())[:10]
        lines.append("")
        lines.append("| Key | Type | Sample value |")
        lines.append("|-----|------|--------------|")
        for k, v in sample_items:
            vtype = _json_shape(v).split("\n")[0][:40]
            vsample = str(v)[:MAX_CELL_CHARS] if not isinstance(v, (dict, list)) else ""
            lines.append(f"| {k} | {vtype} | {_truncate_cell(vsample)} |")
        if n_keys > 10:
            lines.append(f"| _… {n_keys - 10} more keys_ | | |")

    elif isinstance(data, list):
        lines.append(f"JSON array with **{len(data)}** items.")
        if data:
            lines.append("")
            lines.append("First item shape:")
            lines.append("")
            lines.append("```")
            lines.append(_json_shape(data[0]))
            lines.append("```")

    else:
        lines.append(f"JSON {type(data).__name__}: {data}")

    lines.append("")
    return "\n".join(lines)
