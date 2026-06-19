"""Configuration loader — reads config.toml into typed pydantic models.

Uses stdlib tomllib (Python 3.11+). Mirrors every section/field in config.toml
exactly. See §9 of ARCHITECTURE.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

# -- section models ----------------------------------------------------------


class OpenRouterCfg(BaseModel):
    base_url: str
    api_key: str


class ModelsCfg(BaseModel):
    text: str
    vision: str
    embedding: str
    stt: str
    chat: str
    judge: str


class IngestionCfg(BaseModel):
    merge_threshold: float
    pdf_dpi: int
    pdf_image_format: str
    pdf_alpha: bool
    vision_max_images_per_request: int
    vision_max_edge_px: int
    max_audio_minutes: int


class TypesCfg(BaseModel):
    text: list[str]
    code: list[str]
    structured: list[str]
    vision: list[str]
    pdf: list[str]
    office: list[str]
    web: list[str]
    ebook: list[str]
    audio: list[str]
    video: list[str]


class PrivacyCfg(BaseModel):
    zdr: bool
    block_training_providers: bool
    sensitive_patterns: list[str]
    api_key_source: str


class ExtractionCfg(BaseModel):
    primary_model: str
    repair_model: str
    max_attempts: int
    enable_healing: bool
    require_parameters: bool
    confidence_floor: float
    deadletter_dir: str
    quarantine_dir: str


class EvalCfg(BaseModel):
    sample_chat_faithfulness: float
    sample_merge_reversibility: float
    cost_alert_daily_usd: float
    golden_set_dir: str


class GitCfg(BaseModel):
    enabled: bool
    ignore_inbox: bool
    commit_on_compaction: bool


# -- top-level config --------------------------------------------------------


class Config(BaseModel):
    """Top-level config matching config.toml, plus resolved brain_root."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    openrouter: OpenRouterCfg
    models: ModelsCfg
    ingestion: IngestionCfg
    types: TypesCfg
    privacy: PrivacyCfg
    extraction: ExtractionCfg
    eval: EvalCfg
    git: GitCfg
    brain_root: Path


# -- helpers -----------------------------------------------------------------


def _find_config_toml(start: Path | None = None) -> Path:
    """Walk upward from *start* (default: cwd) to find config.toml.

    Falls back to searching upward from this module's parent directories.
    """
    if start is None:
        start = Path.cwd()

    # Search upward from the starting path
    for parent in [start] + list(start.parents):
        candidate = parent / "config.toml"
        if candidate.is_file():
            return candidate.resolve()

    # Fallback: search upward from this file's location
    module_dir = Path(__file__).resolve().parent
    for parent in [module_dir] + list(module_dir.parents):
        candidate = parent / "config.toml"
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "config.toml not found. Run from the second-brain repo root, "
        "or ensure config.toml is in a parent directory."
    )


def load_config(path: Path | None = None) -> Config:
    """Parse config.toml and return a validated Config instance.

    Args:
        path: Optional starting directory to search upward from.
              Defaults to the current working directory, then the module path.

    Returns:
        A validated Config model.

    Raises:
        FileNotFoundError: if config.toml cannot be found.
    """
    config_path = _find_config_toml(path)
    brain_root = config_path.parent.resolve()
    raw = config_path.read_bytes()
    data: dict[str, Any] = tomllib.loads(raw.decode("utf-8"))
    data["brain_root"] = brain_root
    return Config.model_validate(data)


def ext_to_stage(ext: str, types: TypesCfg) -> str:
    """Return the pipeline stage name for a file extension.

    Args:
        ext: File extension, with or without leading dot (case-insensitive).
        types: A TypesCfg instance.

    Returns:
        The stage name (e.g. "text", "pdf", "vision").

    Raises:
        ValueError: if the extension is not registered in *types*.
    """
    ext = ext.lstrip(".").lower()
    for field_name in TypesCfg.model_fields:
        ext_list = getattr(types, field_name)
        if ext in ext_list:
            return field_name
    raise ValueError(f"Unknown extension: .{ext}")
