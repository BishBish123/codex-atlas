"""Generate an asciicast v2 demo file for codex-atlas.

Captures real stdout from each CLI command and assembles it into a valid
asciicast v2 JSON file at assets/demo.cast.

Usage::

    uv run python scripts/generate_demo_cast.py

The resulting file can be played back with::

    asciinema play assets/demo.cast

or via the helper wrapper::

    bash scripts/play_demo.sh
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
CAST_PATH = ASSETS_DIR / "demo.cast"

# ANSI colour helpers (keeps the terminal output readable in the cast)
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RESET = "\033[0m"
_PROMPT = f"{_GREEN}${_RESET} "


# ---------------------------------------------------------------------------
# Asciicast helpers
# ---------------------------------------------------------------------------


def _header(width: int = 120, height: int = 40) -> dict[str, Any]:
    """Return an asciicast v2 header object."""
    return {
        "version": 2,
        "width": width,
        "height": height,
        "timestamp": int(time.time()),
        "title": "codex-atlas demo",
        "env": {"TERM": "xterm-256color"},
    }


def _events_from_command(
    cmd_display: str,
    output: str,
    t_start: float,
    char_delay: float = 0.04,
    line_delay: float = 0.05,
    pre_delay: float = 0.3,
) -> tuple[list[tuple[float, str, str]], float]:
    """Build asciicast events for a single command.

    Returns (events, t_end) where t_end is the timestamp after the last event.

    Events:
    - Prompt appears
    - Command is typed character by character (input events)
    - CRLF to submit
    - Output lines are emitted with small per-line delays (output events)
    """
    events: list[tuple[float, str, str]] = []
    t = t_start + pre_delay

    # Show the prompt
    events.append((t, "o", _PROMPT))
    t += 0.05

    # Type the command character by character
    for ch in cmd_display:
        events.append((t, "i", ch))
        events.append((t, "o", ch))  # terminal echoes input
        t += char_delay

    # Press Enter
    t += 0.1
    events.append((t, "i", "\r\n"))
    events.append((t, "o", "\r\n"))
    t += 0.2

    # Emit output line by line
    for line in output.splitlines():
        events.append((t, "o", line + "\r\n"))
        t += line_delay

    # Blank line after command output
    events.append((t, "o", "\r\n"))
    t += 0.1

    return events, t


def _run_atlas(args: list[str]) -> str:
    """Run an atlas CLI command and return its captured stdout+stderr."""
    cmd = ["uv", "run", "atlas", *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    combined = result.stdout
    if result.returncode != 0:
        raise RuntimeError(
            f"atlas {' '.join(args)} failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return combined


def _run_mcp_client_dry_run() -> str:
    """Run examples/mcp_client.py --dry-run and return captured stdout."""
    cmd = [sys.executable, str(REPO_ROOT / "examples" / "mcp_client.py"), "--dry-run"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mcp_client.py --dry-run failed (exit {result.returncode}):\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _read_eval_headline() -> str:
    """Read the headline section from evals/REPORT.md (written by atlas eval)."""
    report_path = REPO_ROOT / "evals" / "REPORT.md"
    if not report_path.exists():
        return "evals/REPORT.md not found — run: uv run atlas eval"
    text = report_path.read_text(encoding="utf-8")
    # Return only up to (not including) the "## By category" section
    idx = text.find("## By category")
    if idx != -1:
        return text[:idx].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate() -> None:
    """Build assets/demo.cast from real subprocess outputs."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- collect real output -----------------------------------------------
    print("Collecting real demo output...", flush=True)

    index_output = _run_atlas(["index", "src/", "--store=memory"])
    print(f"  index: {index_output.strip()!r}", flush=True)

    ask_output = _run_atlas(["ask", "who calls Agent.run", "--store=memory"])
    print("  ask: ok", flush=True)

    search_output = _run_atlas(
        ["search", "retrieval logic", "--store=memory", "--top-k", "3"]
    )
    print("  search: ok", flush=True)

    explain_output = _run_atlas(
        ["explain", "codex_atlas.retriever.Retriever.retrieve"]
    )
    print("  explain: ok", flush=True)

    eval_output = _read_eval_headline()
    print("  eval (from REPORT.md): ok", flush=True)

    mcp_output = _run_mcp_client_dry_run()
    print("  mcp_client --dry-run: ok", flush=True)

    # ---- build event stream ------------------------------------------------
    all_events: list[tuple[float, str, str]] = []
    t: float = 0.5  # initial pause

    # Section header
    all_events.append(
        (
            t,
            "o",
            f"{_BOLD}{_CYAN}=== codex-atlas interactive demo ==={_RESET}\r\n\r\n",
        )
    )
    t += 0.3

    # 1. index
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- index src/ into an in-memory store ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        "uv run atlas index src/ --store=memory",
        index_output,
        t,
        char_delay=0.03,
    )
    all_events.extend(evs)

    # 2. ask
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- ask a structural question ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        'uv run atlas ask "who calls Agent.run" --store=memory',
        ask_output,
        t,
        char_delay=0.03,
    )
    all_events.extend(evs)

    # 3. search
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- raw retrieval search, top-3 ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        'uv run atlas search "retrieval logic" --store=memory --top-k 3',
        search_output,
        t,
        char_delay=0.03,
    )
    all_events.extend(evs)

    # 4. explain
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- explain a symbol by qualified name ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        'uv run atlas explain "codex_atlas.retriever.Retriever.retrieve"',
        explain_output,
        t,
        char_delay=0.025,
    )
    all_events.extend(evs)

    # 5. eval headline
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- eval harness headline (from evals/REPORT.md) ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        "uv run atlas eval",
        eval_output,
        t,
        char_delay=0.025,
        line_delay=0.04,
    )
    all_events.extend(evs)

    # 6. mcp_client --dry-run
    all_events.append(
        (
            t,
            "o",
            f"{_YELLOW}# --- MCP client example (--dry-run) ---{_RESET}\r\n",
        )
    )
    t += 0.2
    evs, t = _events_from_command(
        "uv run python examples/mcp_client.py --dry-run",
        mcp_output,
        t,
        char_delay=0.025,
    )
    all_events.extend(evs)

    # Final prompt
    all_events.append((t, "o", _PROMPT))

    # ---- write asciicast v2 file -------------------------------------------
    header = _header()
    lines = [json.dumps(header)]
    for ts, event_type, text in all_events:
        lines.append(json.dumps([round(ts, 6), event_type, text]))

    CAST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {CAST_PATH}  ({len(all_events)} events, duration {t:.1f}s)")


if __name__ == "__main__":
    generate()
