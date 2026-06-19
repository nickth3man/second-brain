"""Phase 0 scaffold tests — no network, no API key needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from second_brain.atomicio import rolling_backup, write_atomic, write_json_atomic
from second_brain.config import Config, TypesCfg, ext_to_stage, load_config
from second_brain.slug import slugify

# ── slugify ─────────────────────────────────────────────────────────────────


class TestSlugify:
    """Verify the §10 slug rule against the worked examples."""

    def test_the_tourist_trap(self) -> None:
        assert slugify("The Tourist Trap") == "the-tourist-trap"

    def test_barrel(self) -> None:
        assert slugify("Barrel (The Tourist Trap)") == "barrel-the-tourist-trap"

    def test_members_npcs(self) -> None:
        assert slugify("Members' NPCs") == "members-npcs"


# ── write_atomic ────────────────────────────────────────────────────────────


class TestWriteAtomic:
    def test_writes_and_cleans_tmp(self, tmp_path: Path) -> None:
        target = tmp_path / "hello.txt"
        write_atomic(target, "Hello, world!")
        assert target.read_text() == "Hello, world!"
        # Temp file should be gone
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_bytes_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "data.bin"
        write_atomic(target, b"\x00\x01\x02", text_mode=False)
        assert target.read_bytes() == b"\x00\x01\x02"
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "nested" / "file.txt"
        write_atomic(target, "nested content")
        assert target.read_text() == "nested content"


class TestWriteJsonAtomic:
    def test_writes_json(self, tmp_path: Path) -> None:
        target = tmp_path / "data.json"
        obj = {"a": 1, "b": [2, 3]}
        write_json_atomic(target, obj)
        assert json.loads(target.read_text()) == obj


# ── rolling_backup ──────────────────────────────────────────────────────────


class TestRollingBackup:
    def test_creates_bak_chain(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        p.write_text("v1")

        rolling_backup(p, depth=3)
        assert (tmp_path / "state.json.bak").read_text() == "v1"

        p.write_text("v2")
        rolling_backup(p, depth=3)
        assert (tmp_path / "state.json.bak").read_text() == "v2"
        assert (tmp_path / "state.json.bak-1").read_text() == "v1"

        p.write_text("v3")
        rolling_backup(p, depth=3)
        assert (tmp_path / "state.json.bak").read_text() == "v3"
        assert (tmp_path / "state.json.bak-1").read_text() == "v2"
        assert (tmp_path / "state.json.bak-2").read_text() == "v1"

    def test_caps_at_depth(self, tmp_path: Path) -> None:
        p = tmp_path / "state.json"
        for v in ("v1", "v2", "v3", "v4"):
            p.write_text(v)
            rolling_backup(p, depth=3)
        # Verify depth is capped at 3 (no .bak-3)
        bak_files = sorted(p.parent.glob("state.json.bak*"))
        assert len(bak_files) == 3
        assert not (tmp_path / "state.json.bak-3").exists()

    def test_noop_when_missing(self, tmp_path: Path) -> None:
        p = tmp_path / "nonexistent.json"
        rolling_backup(p, depth=3)
        assert not (tmp_path / "nonexistent.json.bak").exists()


# ── ext_to_stage ────────────────────────────────────────────────────────────


@pytest.fixture
def sample_types() -> TypesCfg:
    return TypesCfg(
        text=["md", "txt"],
        code=["py", "js"],
        structured=["json", "yaml"],
        vision=["png", "jpg"],
        pdf=["pdf"],
        office=["docx"],
        web=["html"],
        ebook=["epub"],
        audio=["mp3"],
        video=["mp4"],
    )


class TestExtToStage:
    def test_known_extensions(self, sample_types: TypesCfg) -> None:
        assert ext_to_stage("md", sample_types) == "text"
        assert ext_to_stage(".md", sample_types) == "text"
        assert ext_to_stage(".PDF", sample_types) == "pdf"
        assert ext_to_stage("json", sample_types) == "structured"
        assert ext_to_stage("mp4", sample_types) == "video"

    def test_unknown_extension(self, sample_types: TypesCfg) -> None:
        with pytest.raises(ValueError, match="Unknown extension"):
            ext_to_stage(".xyz", sample_types)


# ── load_config ─────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_loads_repo_config(self) -> None:
        """Load the repo's own config.toml and verify key values."""
        cfg = load_config()
        assert isinstance(cfg, Config)
        assert cfg.models.text == "anthropic/claude-sonnet-4.5"
        assert cfg.privacy.zdr is True
        assert cfg.brain_root.exists()
        assert (cfg.brain_root / "config.toml").exists()
