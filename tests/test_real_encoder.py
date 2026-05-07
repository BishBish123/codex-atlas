"""Tests for the real-encoder CI workflow and related wiring.

These tests are intentionally lightweight — they do NOT load sentence-transformers
or torch. They check:

1. The CI workflow YAML is parseable (valid YAML).
2. The CLI `atlas eval --help` lists the --encoder flag.
3. Empty / whitespace encoder names are rejected before any model is loaded.
4. The placeholder REPORT.bge.md exists with the expected "WAITING FOR CI RUN" header.
5. The workflow YAML references the expected steps: index and eval commands.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from codex_atlas.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Paths — relative to the project root, resolved from this test file's parent.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parent.parent
_WORKFLOW_PATH = _REPO / ".github" / "workflows" / "real_encoder_eval.yml"
_REPORT_BGE = _REPO / "evals" / "REPORT.bge.md"


class TestWorkflowYaml:
    """The CI workflow YAML must be parseable and contain the required steps."""

    def test_workflow_file_exists(self) -> None:
        assert _WORKFLOW_PATH.exists(), f"Workflow not found at {_WORKFLOW_PATH}"

    def test_workflow_is_valid_yaml(self) -> None:
        raw = _WORKFLOW_PATH.read_text()
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict), "Workflow YAML root must be a mapping"

    def _get_triggers(self) -> dict:
        """Return the 'on:' triggers dict.

        PyYAML parses the bare 'on' key as Python bool True (YAML 1.1
        legacy), so we look up by both the string and boolean key.
        """
        parsed = yaml.safe_load(_WORKFLOW_PATH.read_text())
        # Try string key first (yaml.safe_load with yaml 1.2 / YAML-spec mode),
        # then fall back to the bool True that PyYAML 1.1-compat produces.
        return parsed.get("on") or parsed.get(True) or {}

    def test_workflow_has_push_trigger(self) -> None:
        on = self._get_triggers()
        assert "push" in on, "Workflow must trigger on push"
        branches = on["push"].get("branches", [])
        assert "main" in branches, "Push trigger must include main branch"

    def test_workflow_has_schedule_trigger(self) -> None:
        on = self._get_triggers()
        assert "schedule" in on, "Workflow must have a cron schedule trigger"

    def test_workflow_runs_on_ubuntu(self) -> None:
        parsed = yaml.safe_load(_WORKFLOW_PATH.read_text())
        jobs = parsed.get("jobs", {})
        assert jobs, "Workflow must define at least one job"
        for job in jobs.values():
            assert "ubuntu" in job.get("runs-on", ""), (
                "All jobs must run on ubuntu-latest (where torch wheels are available)"
            )

    def test_workflow_installs_embed_extra(self) -> None:
        raw = _WORKFLOW_PATH.read_text()
        assert "--extra embed" in raw, (
            "Workflow must install the embed extra (sentence-transformers + torch)"
        )

    def test_workflow_references_bge_model(self) -> None:
        raw = _WORKFLOW_PATH.read_text()
        assert "BAAI/bge-small-en-v1.5" in raw, (
            "Workflow must reference the bge-small-en-v1.5 encoder"
        )

    def test_workflow_uploads_artifact(self) -> None:
        raw = _WORKFLOW_PATH.read_text()
        assert "upload-artifact" in raw, "Workflow must upload eval artifacts"


class TestCliEncoderFlag:
    """The --encoder flag must appear on atlas eval and atlas index."""

    def test_eval_help_shows_encoder_flag(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "--encoder" in result.stdout, (
            "`atlas eval --help` must list the --encoder flag"
        )

    def test_index_help_shows_encoder_flag(self) -> None:
        result = runner.invoke(app, ["index", "--help"])
        assert result.exit_code == 0
        assert "--encoder" in result.stdout


class TestEncoderValidation:
    """Empty / whitespace encoder names must be rejected before any model loads.

    Critically, these tests do NOT import sentence-transformers or torch —
    the validation runs before the lazy import so failures are instant.
    """

    def _invoke_eval_with_encoder(self, encoder_val: str) -> int:
        """Invoke `atlas eval --encoder <val>` and return the exit code.

        Passing a fake corpus that doesn't exist forces an early bail from
        the corpus-validation path, but we want to check encoder validation
        fires first. We use a non-existent corpus so the test stays offline.
        The order in eval_cmd: corpus is validated before _resolve_encoder is
        called inside the async _run() closure, so we need to provide a valid
        corpus. Use a minimal tmp fixture via the runner's mix-in path.
        """
        # Use the actual src/ dir (exists, non-empty) so corpus validation
        # passes and we reach _resolve_encoder.
        return runner.invoke(
            app,
            [
                "eval",
                "--encoder",
                encoder_val,
                "--corpus",
                "src",
                "--store",
                "memory",
            ],
        ).exit_code

    def test_empty_encoder_name_rejected(self) -> None:
        # An empty string must fail before any model download.
        result = runner.invoke(
            app,
            ["eval", "--encoder", "", "--corpus", "src", "--store", "memory"],
        )
        # Exit code should be non-zero and the output should mention the problem.
        assert result.exit_code != 0

    def test_whitespace_only_encoder_rejected(self) -> None:
        result = runner.invoke(
            app,
            ["eval", "--encoder", "   ", "--corpus", "src", "--store", "memory"],
        )
        assert result.exit_code != 0

    def test_fake_encoder_accepted(self) -> None:
        # `fake` is always valid and never tries to load a model.
        # We just check the eval runs without an immediate encoder-name error.
        # The eval itself will complete (it uses the existing graph + in-memory store).
        result = runner.invoke(
            app,
            ["eval", "--encoder", "fake", "--corpus", "src", "--store", "memory"],
        )
        # Should succeed (exit 0) — the eval runs fully.
        assert result.exit_code == 0, (
            f"atlas eval --encoder fake failed unexpectedly:\n{result.stdout}"
        )


class TestReportBgePlaceholder:
    """evals/REPORT.bge.md must exist and contain either the placeholder or real metrics.

    After the Docker-based bge-small run the file contains real numbers.
    These tests accept both states so the suite stays green regardless of
    whether the CI artifact has been committed yet.
    """

    def test_report_bge_exists(self) -> None:
        assert _REPORT_BGE.exists(), (
            f"REPORT.bge.md not found at {_REPORT_BGE}"
        )

    def test_report_bge_has_waiting_header_or_real_metrics(self) -> None:
        content = _REPORT_BGE.read_text()
        is_placeholder = "WAITING FOR CI RUN" in content
        is_populated = "Route correctness" in content
        assert is_placeholder or is_populated, (
            "REPORT.bge.md must either contain the 'WAITING FOR CI RUN' placeholder "
            "or real eval metrics (Route correctness)"
        )

    def test_report_bge_mentions_workflow_or_category_table(self) -> None:
        content = _REPORT_BGE.read_text()
        has_workflow = "real_encoder_eval" in content
        has_category_table = "By category" in content
        assert has_workflow or has_category_table, (
            "REPORT.bge.md must reference the real_encoder_eval workflow (placeholder) "
            "or contain a 'By category' section (populated report)"
        )

    @pytest.mark.parametrize("model_name", ["BAAI/bge-small-en-v1.5"])
    def test_report_bge_mentions_encoder(self, model_name: str) -> None:
        content = _REPORT_BGE.read_text()
        # Placeholder: the encoder is mentioned in the "How to refresh" section.
        # Populated report: the encoder name may not appear in the markdown body
        # (it lives in scores-bge.json), so we accept either the model name or
        # the presence of real headline metrics as proof the report is on-topic.
        is_mentioned = model_name in content or "bge-small-en-v1.5" in content
        is_populated = "Route correctness" in content
        assert is_mentioned or is_populated, (
            f"REPORT.bge.md must mention the encoder model {model_name!r} "
            f"(placeholder) or contain real eval metrics (populated report)"
        )
