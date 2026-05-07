#!/usr/bin/env bash
# Play assets/demo.cast with asciinema.
# If asciinema is not installed, prints an install hint and exits 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAST_FILE="$REPO_ROOT/assets/demo.cast"

if ! command -v asciinema &> /dev/null; then
    echo "asciinema is not installed or not on PATH."
    echo ""
    echo "Install it with:"
    echo "  macOS (Homebrew):  brew install asciinema"
    echo "  pip:               pip install asciinema"
    echo "  apt:               sudo apt install asciinema"
    echo ""
    echo "Once installed, run:  asciinema play $CAST_FILE"
    exit 1
fi

if [[ ! -f "$CAST_FILE" ]]; then
    echo "Cast file not found: $CAST_FILE"
    echo "Generate it first with:  make demo-cast"
    exit 1
fi

exec asciinema play "$CAST_FILE"
