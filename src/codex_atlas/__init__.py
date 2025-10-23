"""Codex-Atlas: agentic GraphRAG over a codebase, exposed as an MCP server."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("codex-atlas")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
