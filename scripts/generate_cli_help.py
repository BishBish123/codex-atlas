"""Generate docs/CLI.md from ``atlas --help`` and each subcommand's ``--help`` output.

Usage::

    uv run python scripts/generate_cli_help.py

The resulting file is committed at docs/CLI.md and can be regenerated at any
time via::

    make cli-help

Design notes
------------
* Uses ``subprocess.run`` with ``capture_output=True`` so nothing leaks to the
  terminal during generation.
* Discovers subcommands by parsing the top-level ``atlas --help`` output — no
  hard-coded list means new subcommands appear automatically.
* Each section includes: command synopsis, the raw help text in a fenced code
  block, and a one-line description pulled from the first non-empty line of the
  command's help output.
* Exits non-zero (and prints a warning) if a subcommand's ``--help`` fails,
  but continues processing the remaining subcommands so the document is always
  as complete as possible.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_PATH = DOCS_DIR / "CLI.md"

# The CLI entry-point as installed by uv / the project scripts table.
CLI_ENTRY = ["uv", "run", "atlas"]

# Description overrides: keys are subcommand names, values replace the
# auto-extracted one-liner.  Extend here for subcommands whose first help line
# is a usage synopsis rather than a description.
DESCRIPTION_OVERRIDES: dict[str, str] = {}


def run_help(args: list[str]) -> tuple[bool, str]:
    """Run ``args + ["--help"]`` and return (success, stdout_text)."""
    result = subprocess.run(
        [*args, "--help"],
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    output = result.stdout if ok else result.stderr or result.stdout
    return ok, output


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return re.sub(r"\x1b\[[0-9;]*[mKJH]", "", text)


def extract_subcommands(help_text: str) -> list[str]:
    """Parse subcommand names from the top-level ``--help`` output.

    Expects a Rich/Typer Commands panel.  Each first-column command entry
    is a line of the form ``│ <name>  <description>``.  Continuation lines
    (word-wrapped description) start with ``│            `` (many spaces)
    with no command-like token at the same indent.

    We identify command lines by requiring the token to appear at a
    consistent left indent: exactly ``│ `` (pipe + one space) followed
    immediately by the command name with no leading spaces.
    """
    names: list[str] = []
    in_commands = False
    for line in help_text.splitlines():
        # Rich panel header — "╭─ Commands ───"
        if re.search(r"Commands", line, re.IGNORECASE):
            in_commands = True
            continue
        if in_commands:
            # End of the Commands panel
            if line.strip().startswith("╰"):
                break
            # Command-name lines: "│ <name>   <description>"
            # The name starts right after "│ " (pipe + single space).
            match = re.match(r"^│ ([a-z][a-z0-9_-]*)\s{2,}", line)
            if match:
                names.append(match.group(1))
    return names


def extract_one_liner(help_text: str, subcommand: str) -> str:
    """Return a short description for *subcommand*.

    Looks for the first non-blank line after the ``Usage:`` line — that is
    typically the docstring / summary Typer surfaces.  Falls back to the
    subcommand name if nothing useful is found.
    """
    if subcommand in DESCRIPTION_OVERRIDES:
        return DESCRIPTION_OVERRIDES[subcommand]

    lines = [l.strip().lstrip("│ \t") for l in help_text.splitlines()]
    past_usage = False
    for line in lines:
        if line.lower().startswith("usage:"):
            past_usage = True
            continue
        if past_usage and line and not line.startswith("╭") and not line.startswith("│"):
            # Skip option/argument panel headers
            return line
    return subcommand


def build_section(subcommand: str, top_help_line: str) -> str:
    """Build the markdown section for *subcommand*.

    *top_help_line* is the brief description from the top-level help table
    (used as a fallback when the subcommand help is unavailable).
    """
    ok, raw_help = run_help([*CLI_ENTRY, subcommand])
    clean_help = strip_ansi(raw_help)

    if not ok:
        print(
            f"  WARNING: `atlas {subcommand} --help` exited non-zero — including partial output.",
            file=sys.stderr,
        )
        one_liner = top_help_line or f"(help unavailable for {subcommand!r})"
    else:
        one_liner = extract_one_liner(clean_help, subcommand) or top_help_line

    synopsis = f"atlas {subcommand} [OPTIONS]"

    lines = [
        f"## `atlas {subcommand}`",
        "",
        f"_{one_liner}_",
        "",
        f"**Synopsis:** `{synopsis}`",
        "",
        "```",
        clean_help.rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def extract_top_descriptions(help_text: str) -> dict[str, str]:
    """Return mapping of subcommand name → brief description from top-level help."""
    result: dict[str, str] = {}
    in_commands = False
    for line in help_text.splitlines():
        if re.search(r"Commands", line, re.IGNORECASE):
            in_commands = True
            continue
        if in_commands:
            if line.strip().startswith("╰"):
                break
            # Same pattern as extract_subcommands — strict left indent
            match = re.match(r"^│ ([a-z][a-z0-9_-]*)\s{2,}(.*)", line)
            if match:
                result[match.group(1)] = match.group(2).strip()
    return result


def generate(output_path: Path = OUTPUT_PATH) -> str:
    """Generate the CLI reference markdown and return it as a string.

    Also writes the result to *output_path*.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ok, top_help_raw = run_help(CLI_ENTRY)
    if not ok:
        print(
            "ERROR: `atlas --help` failed — is the package installed?",
            file=sys.stderr,
        )
        sys.exit(1)

    top_help = strip_ansi(top_help_raw)
    subcommands = extract_subcommands(top_help)
    top_descriptions = extract_top_descriptions(top_help)

    header = "\n".join(
        [
            "# Atlas CLI Reference",
            "",
            "> Auto-generated by `scripts/generate_cli_help.py`. "
            "Run `make cli-help` to refresh.",
            "",
            "## Top-level",
            "",
            "```",
            top_help.rstrip(),
            "```",
            "",
            "---",
            "",
        ]
    )

    sections: list[str] = [header]

    print(f"Found {len(subcommands)} subcommands: {', '.join(subcommands)}")
    for sub in subcommands:
        print(f"  Generating section for: atlas {sub}")
        sections.append(build_section(sub, top_descriptions.get(sub, "")))
        sections.append("---\n")

    # atlas-mcp entry-point (separate binary, not a subcommand)
    mcp_section = build_atlas_mcp_section()
    if mcp_section:
        sections.append(mcp_section)
        sections.append("---\n")

    content = "\n".join(sections)
    output_path.write_text(content, encoding="utf-8")
    print(f"Written to {output_path}")
    return content


def build_atlas_mcp_section() -> str:
    """Build a section for the ``atlas-mcp`` entry-point binary."""
    result = subprocess.run(
        ["uv", "run", "atlas-mcp", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""

    clean = strip_ansi(result.stdout)
    lines = [
        "## `atlas-mcp` (MCP server entry-point)",
        "",
        "_Standalone entry-point that starts the Codex-Atlas MCP server directly "
        "(equivalent to `atlas mcp`)._",
        "",
        "**Synopsis:** `atlas-mcp [OPTIONS]`",
        "",
        "```",
        clean.rstrip(),
        "```",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    generate()
