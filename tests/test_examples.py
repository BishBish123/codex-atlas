"""Tests for examples/mcp_client.py.

We exercise the formatter functions and the --dry-run code path without
spawning a real stdio server — that would be flaky in CI and require
data/graph.json to exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Make ``examples/`` importable from the tests directory.
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

import mcp_client


# ---------------------------------------------------------------------------
# Formatter unit tests
# ---------------------------------------------------------------------------


def test_format_search_result_happy_path() -> None:
    data: dict[str, Any] = {
        "route": "structural",
        "grade": 0.85,
        "attempts": 1,
        "answer": "Agent.run is the entry point.",
        "citations": [
            {
                "qualified_name": "codex_atlas.agent.Agent.run",
                "file_path": "src/codex_atlas/agent.py",
                "lineno_start": 500,
                "lineno_end": 540,
                "score": 0.91,
                "text": "async def run ...",
            }
        ],
        "cancelled": None,
    }
    out = mcp_client.format_search_result("search_code", data)
    assert "[search_code]" in out
    assert "route" in out
    assert "structural" in out
    assert "0.85" in out
    assert "codex_atlas.agent.Agent.run" in out


def test_format_search_result_timeout() -> None:
    data: dict[str, Any] = {"error": "agent_timeout", "phase": "retrieve"}
    out = mcp_client.format_search_result("search_code", data)
    assert "agent_timeout" in out
    assert "retrieve" in out


def test_format_callers_result() -> None:
    data: dict[str, Any] = {
        "target": "codex_atlas.agent.Agent.run",
        "depth": 1,
        "callers": [
            {"qualified_name": "codex_atlas.mcp_server.search_code"},
            {"qualified_name": "codex_atlas.cli.ask"},
        ],
    }
    out = mcp_client.format_callers_result("find_callers", data)
    assert "find_callers" in out
    assert "codex_atlas.agent.Agent.run" in out
    assert "codex_atlas.mcp_server.search_code" in out
    assert "callers : 2" in out


def test_format_neighborhood_result() -> None:
    data: dict[str, Any] = {
        "target": "codex_atlas.retriever.Retriever.retrieve",
        "depth": 2,
        "callers": ["codex_atlas.agent.Agent._retrieve"],
        "callees": ["codex_atlas.store.ChunkStoreProtocol.search"],
        "all": [
            "codex_atlas.agent.Agent._retrieve",
            "codex_atlas.store.ChunkStoreProtocol.search",
        ],
    }
    out = mcp_client.format_neighborhood_result("get_graph_neighborhood", data)
    assert "get_graph_neighborhood" in out
    assert "codex_atlas.retriever.Retriever.retrieve" in out
    assert "Agent._retrieve" in out


def test_truncate_short() -> None:
    assert mcp_client._truncate("hello", 120) == "hello"


def test_truncate_long() -> None:
    long_text = "x" * 200
    result = mcp_client._truncate(long_text, 120)
    assert len(result) == 120
    assert result.endswith("...")


# ---------------------------------------------------------------------------
# --dry-run end-to-end (no server spawn)
# ---------------------------------------------------------------------------


def test_dry_run_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = mcp_client.main(["--dry-run"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "search_code" in captured.out
    assert "find_callers" in captured.out
    assert "get_graph_neighborhood" in captured.out


def test_dry_run_json_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = mcp_client.main(["--dry-run", "--json"])
    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "search_code" in parsed
    assert "find_callers" in parsed
    assert "get_graph_neighborhood" in parsed


def test_dry_run_search_fixture_has_citations(capsys: pytest.CaptureFixture[str]) -> None:
    """Fixture data must have at least one citation so the formatter output is non-trivial."""
    exit_code = mcp_client.main(["--dry-run"])
    assert exit_code == 0
    captured = capsys.readouterr()
    # The formatter prints the citation qualified name for the first 3 citations.
    assert "codex_atlas.agent.Agent.run" in captured.out
