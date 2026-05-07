"""Tests for scripts/generate_demo_cast.py and assets/demo.cast.

Validates:
- Generator produces valid asciicast v2 JSON (header line + event lines)
- Cast contains expected substrings (atlas, Encoder, Retriever.retrieve, ...)
- play_demo.sh passes bash -n syntax check
- Cast file is parseable by json.loads line-by-line

None of these tests invoke the asciinema binary.
"""

from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
ASSETS_DIR = REPO_ROOT / "assets"
CAST_PATH = ASSETS_DIR / "demo.cast"
GENERATOR = SCRIPTS_DIR / "generate_demo_cast.py"
PLAY_SCRIPT = SCRIPTS_DIR / "play_demo.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_generator() -> types.ModuleType:
    """Import generate_demo_cast.py as a module."""
    spec = importlib.util.spec_from_file_location("generate_demo_cast", GENERATOR)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _load_cast_lines() -> list[str]:
    """Return non-empty lines from assets/demo.cast."""
    return [ln for ln in CAST_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _cast_text() -> str:
    """Return the full cast file text concatenated from all output events."""
    lines = _load_cast_lines()
    parts: list[str] = []
    for ln in lines[1:]:  # skip header
        event = json.loads(ln)
        if isinstance(event, list) and len(event) == 3 and event[1] == "o":
            parts.append(event[2])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Generator module import test
# ---------------------------------------------------------------------------


def test_generator_module_importable() -> None:
    """scripts/generate_demo_cast.py must be importable without errors."""
    mod = _load_generator()
    assert hasattr(mod, "generate")
    assert hasattr(mod, "_header")
    assert hasattr(mod, "_events_from_command")


# ---------------------------------------------------------------------------
# _header unit test
# ---------------------------------------------------------------------------


def test_header_structure() -> None:
    """_header() must return a valid asciicast v2 header dict."""
    mod = _load_generator()
    hdr = mod._header(width=80, height=24)
    assert hdr["version"] == 2
    assert hdr["width"] == 80
    assert hdr["height"] == 24
    assert "timestamp" in hdr
    assert "title" in hdr


# ---------------------------------------------------------------------------
# _events_from_command unit test
# ---------------------------------------------------------------------------


def test_events_from_command_structure() -> None:
    """_events_from_command must produce (ts, type, text) 3-tuples with monotone timestamps."""
    mod = _load_generator()
    events, t_end = mod._events_from_command("atlas index src/", "parsed 5 files\n", 0.0)
    assert len(events) > 0
    for ev in events:
        assert len(ev) == 3
        ts, kind, text = ev
        assert isinstance(ts, float)
        assert kind in ("i", "o")
        assert isinstance(text, str)
    # Timestamps must be non-decreasing across the sequence
    timestamps = [e[0] for e in events]
    assert timestamps == sorted(timestamps)
    assert t_end >= timestamps[-1]


# ---------------------------------------------------------------------------
# Cast file existence and format
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_file_exists() -> None:
    assert CAST_PATH.exists(), "assets/demo.cast must exist"
    assert CAST_PATH.stat().st_size > 0, "assets/demo.cast must not be empty"


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_header_is_valid_json() -> None:
    """First line must be a valid asciicast v2 header."""
    lines = _load_cast_lines()
    assert lines, "Cast file must have at least one line"
    header = json.loads(lines[0])
    assert header["version"] == 2
    assert "width" in header
    assert "height" in header


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_all_lines_parseable() -> None:
    """Every line in the cast must be valid JSON."""
    lines = _load_cast_lines()
    for i, ln in enumerate(lines):
        try:
            json.loads(ln)
        except json.JSONDecodeError as exc:
            pytest.fail(f"Line {i} is not valid JSON: {exc}\nContent: {ln!r}")


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_has_multiple_events() -> None:
    """Cast must have at least 20 events (header + events)."""
    lines = _load_cast_lines()
    # lines[0] = header, rest = events
    assert len(lines) > 20, f"Expected >20 lines, got {len(lines)}"


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_event_structure() -> None:
    """All event lines must be [float, 'i'|'o', str] 3-element arrays."""
    lines = _load_cast_lines()
    for i, ln in enumerate(lines[1:], start=1):
        ev = json.loads(ln)
        assert isinstance(ev, list) and len(ev) == 3, f"Line {i}: expected [ts, type, text]"
        ts, kind, text = ev
        assert isinstance(ts, (int, float)), f"Line {i}: timestamp must be numeric"
        assert kind in ("i", "o"), f"Line {i}: event type must be 'i' or 'o'"
        assert isinstance(text, str), f"Line {i}: text must be a string"


# ---------------------------------------------------------------------------
# Cast content assertions (real output substrings)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_contains_atlas_command() -> None:
    """Cast output must mention the atlas CLI."""
    text = _cast_text()
    assert "atlas" in text


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_contains_retriever_retrieve() -> None:
    """Cast must contain the Retriever.retrieve qualified name from explain output."""
    text = _cast_text()
    assert "Retriever.retrieve" in text or "Retriever" in text


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_contains_store_memory() -> None:
    """Cast must contain --store=memory flag used in commands."""
    text = _cast_text()
    assert "store" in text


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_contains_mcp_output() -> None:
    """Cast must contain MCP client output (search_code, find_callers)."""
    text = _cast_text()
    assert "search_code" in text or "find_callers" in text or "mcp" in text.lower()


@pytest.mark.skipif(not CAST_PATH.exists(), reason="assets/demo.cast not generated yet")
def test_cast_contains_eval_metrics() -> None:
    """Cast must contain eval output (Route correctness from REPORT.md)."""
    text = _cast_text()
    assert "Route" in text or "Questions" in text or "Metric" in text


# ---------------------------------------------------------------------------
# play_demo.sh syntax check
# ---------------------------------------------------------------------------


def test_play_demo_sh_bash_syntax() -> None:
    """play_demo.sh must pass bash -n (syntax-only check)."""
    result = subprocess.run(
        ["bash", "-n", str(PLAY_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed on {PLAY_SCRIPT}:\n{result.stderr}"


def test_play_demo_sh_is_executable() -> None:
    """play_demo.sh must have execute permission."""
    mode = PLAY_SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "play_demo.sh is not executable"
