"""Tests for structured-file summarization.

CSV/JSON summarizers + extract branching + pipeline integration.

Hermetic — no real API calls; uses fakes for client/embedder.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from second_brain.parse.csv_table import summarize_csv, summarize_json

# ---------------------------------------------------------------------------
# summarize_csv
# ---------------------------------------------------------------------------


class TestSummarizeCsv:
    """summarize_csv column/type inference, sample rendering, edge cases."""

    def test_small_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "test.csv"
        csv_file.write_text(
            "name,age,score,active\n"
            "Alice,30,95.5,true\n"
            "Bob,25,88.0,false\n"
            "Charlie,35,92.3,true\n"
        )
        result = summarize_csv(csv_file)
        assert "# test" in result
        assert "3 rows x 4 columns" in result
        assert "name" in result
        assert "age" in result
        assert "score" in result
        assert "active" in result
        # Type inference
        assert "int" in result or "float" in result
        # Sample rows table
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" in result

    def test_data_dictionary_csv(self, tmp_path: Path) -> None:
        """A data-dictionary-style CSV renders sensibly."""
        csv_file = tmp_path / "data_dictionary.csv"
        csv_file.write_text(
            "File,Column,Type,Description\n"
            "games.csv,game_id,int,Unique game identifier\n"
            "games.csv,date,date,Date the game was played\n"
            "games.csv,home_team,string,Home team name\n"
            "games.csv,away_team,string,Away team name\n"
            "games.csv,home_score,int,Points scored by home team\n"
        )
        result = summarize_csv(csv_file)
        assert "# data_dictionary" in result
        assert "5 rows x 4 columns" in result
        assert "File" in result
        assert "Column" in result
        assert "Type" in result
        assert "Description" in result

    def test_empty_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        result = summarize_csv(csv_file)
        assert "Empty file" in result

    def test_csv_with_only_header(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "only_header.csv"
        csv_file.write_text("a,b,c\n")
        result = summarize_csv(csv_file)
        assert "0 rows" in result

    def test_malformed_ragged_rows(self, tmp_path: Path) -> None:
        """Ragged rows don't crash."""
        csv_file = tmp_path / "ragged.csv"
        csv_file.write_text(
            "a,b,c\n"
            "1,2,3\n"
            "4,5\n"
            "6,7,8,9\n"
        )
        result = summarize_csv(csv_file)
        assert "inconsistent column counts" in result
        assert "3 rows" in result

    def test_tsv_file(self, tmp_path: Path) -> None:
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text(
            "name\tvalue\n"
            "x\t10\n"
            "y\t20\n"
        )
        result = summarize_csv(tsv_file)
        assert "# data" in result
        assert "2 rows x 2 columns" in result

    def test_large_csv_caps_display(self, tmp_path: Path) -> None:
        """A CSV with more than 100k rows caps the count display."""
        csv_file = tmp_path / "large.csv"
        # Write header + 100_001 rows (one more than cap)
        lines = ["a,b\n"] + ["1,2\n"] * 100_001
        csv_file.write_text("".join(lines))
        result = summarize_csv(csv_file)
        assert "100,000+" in result

    def test_long_cell_truncation(self, tmp_path: Path) -> None:
        """Cells longer than max_cell_chars are truncated."""
        csv_file = tmp_path / "long.csv"
        csv_file.write_text(
            "note\n"
            + ("x" * 200 + "\n")
        )
        result = summarize_csv(csv_file, max_cell_chars=10)
        assert "..." in result
        assert len([line for line in result.split("\n") if "..." in line]) > 0

    def test_utf8_bom(self, tmp_path: Path) -> None:
        """UTF-8 BOM is stripped."""
        csv_file = tmp_path / "bom.csv"
        csv_file.write_bytes("\ufeffcol\nval\n".encode("utf-8"))
        result = summarize_csv(csv_file)
        assert "col" in result

    def test_unicode_decode_error(self, tmp_path: Path) -> None:
        """Non-UTF-8 bytes are handled via replace."""
        csv_file = tmp_path / "bad_enc.csv"
        csv_file.write_bytes(b"col\n\xff\xfe\n")
        result = summarize_csv(csv_file)
        # Should not crash; replacement chars are fine.
        assert "col" in result

    def test_many_columns(self, tmp_path: Path) -> None:
        """CSV with >40 columns shows grouping note."""
        n_cols = 50
        header = ",".join(f"c{i}" for i in range(n_cols))
        data = ",".join(str(i) for i in range(n_cols))
        csv_file = tmp_path / "wide.csv"
        csv_file.write_text(f"{header}\n{data}\n")
        result = summarize_csv(csv_file)
        # Should mention truncated columns
        assert "more columns" in result
        assert "50 columns" in result

    def test_type_inference_int_float_mixed(self, tmp_path: Path) -> None:
        """Column with ints and floats is typed as float."""
        csv_file = tmp_path / "mixed_num.csv"
        csv_file.write_text("val\n1\n2.5\n3\n")
        result = summarize_csv(csv_file)
        assert "float" in result

    def test_type_inference_bool(self, tmp_path: Path) -> None:
        """Column with true/false values is typed as bool."""
        csv_file = tmp_path / "flags.csv"
        csv_file.write_text("flag\ntrue\nfalse\ntrue\n")
        result = summarize_csv(csv_file)
        assert "bool" in result


# ---------------------------------------------------------------------------
# summarize_json
# ---------------------------------------------------------------------------


class TestSummarizeJson:
    """summarize_json renders shape/key info."""

    def test_json_object(self, tmp_path: Path) -> None:
        jf = tmp_path / "config.json"
        jf.write_text('{"host": "localhost", "port": 8080, "debug": true}')
        result = summarize_json(jf)
        assert "# config" in result
        assert "**3**" in result  # 3 keys
        assert "host" in result
        assert "port" in result
        assert "debug" in result

    def test_json_array(self, tmp_path: Path) -> None:
        jf = tmp_path / "items.json"
        jf.write_text('[{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]')
        result = summarize_json(jf)
        assert "# items" in result
        assert "2" in result  # 2 items

    def test_json_scalar(self, tmp_path: Path) -> None:
        jf = tmp_path / "version.json"
        jf.write_text('"1.2.3"')
        result = summarize_json(jf)
        assert "# version" in result
        assert "str" in result or "string" in result

    def test_json_invalid(self, tmp_path: Path) -> None:
        jf = tmp_path / "bad.json"
        jf.write_text("{invalid")
        result = summarize_json(jf)
        assert "Invalid JSON" in result


# ---------------------------------------------------------------------------
# Extract branching
# ---------------------------------------------------------------------------


class TestExtractBranching:
    """extract uses structured prompt for structured sources."""

    async def test_structured_uses_structured_prompt(self, tmp_path: Path) -> None:
        """Structured source type sends the structured prompt."""
        from second_brain.daemon.extract import build_messages

        msgs = build_messages("col1,col2", {}, source_type="structured")
        sys_content = msgs[0]["content"]
        assert "exactly ONE" in sys_content
        assert "structured/tabular" in sys_content

    async def test_non_structured_uses_default_prompt(self, tmp_path: Path) -> None:
        """Non-structured source type sends the default librarian prompt."""
        from second_brain.daemon.extract import build_messages

        msgs = build_messages("Some text.", {}, source_type="text")
        sys_content = msgs[0]["content"]
        assert "3–7 topics" in sys_content
        assert "structured/tabular" not in sys_content

    async def test_default_prompt_when_no_source_type(self, tmp_path: Path) -> None:
        """Omitting source_type defaults to the librarian prompt."""
        from second_brain.daemon.extract import build_messages

        msgs = build_messages("Some text.", {})
        sys_content = msgs[0]["content"]
        assert "3–7 topics" in sys_content


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """Structured files route through the summarizer in the pipeline."""

    def _make_cfg(self, tmp_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            brain_root=tmp_path,
            models=SimpleNamespace(
                stt="test-stt",
                text="test-text",
                chat="test-chat",
                embedding="test-embed",
            ),
            types=SimpleNamespace(
                text=["txt", "md"],
                code=["py"],
                structured=["csv", "tsv", "json"],
                vision=[],
                pdf=[],
                office=[],
                web=[],
                ebook=[],
                audio=[],
                video=[],
            ),
            ingestion=SimpleNamespace(
                max_audio_minutes=120,
                merge_threshold=0.85,
                vision_max_images_per_request=5,
                require_parameters=False,
                enable_healing=False,
            ),
            extraction=SimpleNamespace(
                deadletter_dir=str(tmp_path / ".brain" / "deadletter"),
                primary_model="",
                repair_model="",
                require_parameters=False,
                enable_healing=False,
            ),
            privacy=SimpleNamespace(
                zdr=True,
                api_key_source="env",
                block_training_providers=False,
            ),
            openrouter=SimpleNamespace(
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-or-v1-test",
            ),
        )

    async def test_structured_file_gets_summarized_body(self, tmp_path: Path) -> None:
        """A CSV file receives a summarized body (not raw CSV) in extract."""
        from second_brain.daemon.pipeline import ingest_file
        from second_brain.models import IngestStage
        from second_brain.state import BrainStateStore

        cfg = self._make_cfg(tmp_path)
        store = BrainStateStore.load(cfg)

        # Create a fake client that captures the extract body
        captured_body: list[str] = []

        class FakeClient:
            async def chat_completion(
                self, model, messages, *,
                response_format=None, extra_body=None, stream=False,
            ):
                # The last message's content is what extract sends
                user_msg = messages[-1]["content"] if messages else ""
                captured_body.append(user_msg)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps({
                                    "tldr": "Test dataset summary",
                                    "topics": [
                                        {
                                            "name": "Test Data",
                                            "action": "new",
                                            "target_slug": "",
                                            "confidence": 0.9,
                                            "merged_section": "A test dataset.",
                                        }
                                    ],
                                })
                            }
                        }
                    ],
                }

            async def chat_completion_clean(
                self, model, messages, *,
                response_format=None, extra_body=None,
            ):
                return (None, "test")

            async def embedding(self, model, input_):
                return [0.1] * 8

            async def transcribe(self, model, audio_path, *, language=None, audio_format=None):
                return "transcript"

            async def close(self):
                pass

        class FakeLinker:
            async def link(self, topics, ctx):
                return []

        class FakeIndex:
            def mark_dirty(self):
                pass
            async def flush_now(self):
                pass

        path = tmp_path / "00-inbox" / "data.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("col1,col2\n1,2\n3,4\n")

        client = FakeClient()
        stage = await ingest_file(
            path, cfg, store, client, FakeLinker(), FakeIndex(),
            embedder=None, vec_store=None,
        )

        assert stage == IngestStage.DONE

        # The extract call received the summarized body, not raw CSV
        assert captured_body
        user_content = captured_body[0]
        assert "col1" in user_content
        assert "2 rows" in user_content  # summarized says 2 rows
        # Raw CSV text "1,2" would appear in the body but as summarized markdown
        # The key check: verify it looks like a summary, not raw CSV
        assert "| col1 |" in user_content
        assert "| col2 |" in user_content

    async def test_structured_extract_receives_source_type(self, tmp_path: Path) -> None:
        """The extract call receives source_type='structured' for CSV files."""
        from second_brain.daemon.extract import build_messages

        # Test the message building path directly
        msgs = build_messages("# data\n\ncol1, col2", {}, source_type="structured")
        assert msgs[0]["content"] != ""
        assert "exactly ONE" in msgs[0]["content"]

    async def test_text_file_unaffected(self, tmp_path: Path) -> None:
        """A .txt file still uses the default librarian prompt and 3-7 topics."""
        from second_brain.daemon.extract import build_messages

        msgs = build_messages("Hello, world.", {}, source_type="text")
        assert "3–7 topics" in msgs[0]["content"]
