#!/usr/bin/env bash
# scripts/deploy.sh — deploy codex-atlas to Fly.io
#
# Usage:
#   ./scripts/deploy.sh                        # uses FLY_APP_NAME env var
#   ./scripts/deploy.sh --app my-codex-atlas   # explicit app name
#
# Required env vars (set via `fly secrets set` or exported locally):
#   FLY_APP_NAME          — Fly.io app name (or pass --app)
#   GROQ_API_KEY          — Groq API key for GroqSynthesizer (optional; uses stitch fallback)
#   ANTHROPIC_API_KEY     — Anthropic key for LLM judge (optional; uses heuristic fallback)
#   OPENAI_API_KEY        — OpenAI key for LLM judge (optional fallback to Anthropic)
#   ATLAS_PG_DSN          — postgresql://user:pass@host/db (required for pgvector store)
#   ATLAS_NEO4J_URI       — bolt://... or neo4j+s://... (required for Neo4j graph backend)
#   ATLAS_NEO4J_USERNAME  — Neo4j username (default: neo4j)
#   ATLAS_NEO4J_PASSWORD  — Neo4j password (required for Neo4j backend)
#   LANGFUSE_PUBLIC_KEY   — Langfuse public key for tracing (optional)
#   LANGFUSE_SECRET_KEY   — Langfuse secret key for tracing (optional)

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
APP_NAME="${FLY_APP_NAME:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)
            APP_NAME="$2"
            shift 2
            ;;
        --app=*)
            APP_NAME="${1#--app=}"
            shift
            ;;
        --help|-h)
            grep '^#' "$0" | head -25 | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

# 1. fly CLI must be on PATH
if ! command -v fly &>/dev/null; then
    echo "ERROR: 'fly' CLI not found on PATH." >&2
    echo "  Install: https://fly.io/docs/hands-on/install-flyctl/" >&2
    exit 1
fi

# 2. Must be authenticated
if ! fly auth whoami &>/dev/null; then
    echo "ERROR: Not logged in to Fly.io." >&2
    echo "  Run: fly auth login" >&2
    exit 1
fi

# 3. App name must be set
if [[ -z "$APP_NAME" ]]; then
    # Try to read from fly.toml in the repo root
    TOML_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
    if [[ -f "$TOML_ROOT/fly.toml" ]]; then
        APP_NAME="$(grep '^app\s*=' "$TOML_ROOT/fly.toml" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')"
    fi
fi

if [[ -z "$APP_NAME" ]] || [[ "$APP_NAME" == *"REPLACE-ME"* ]]; then
    echo "ERROR: App name not set or still contains REPLACE-ME sentinel." >&2
    echo "  Either export FLY_APP_NAME=<name> or pass --app <name>." >&2
    echo "  Also update the 'app' field in fly.toml." >&2
    exit 1
fi

echo "==> Deploying to Fly.io app: ${APP_NAME}"

# ---------------------------------------------------------------------------
# Push secrets from local environment (only the vars that are set)
# ---------------------------------------------------------------------------
SECRETS_ARGS=()

for var in \
    GROQ_API_KEY \
    ANTHROPIC_API_KEY \
    OPENAI_API_KEY \
    ATLAS_PG_DSN \
    ATLAS_NEO4J_URI \
    ATLAS_NEO4J_USERNAME \
    ATLAS_NEO4J_PASSWORD \
    LANGFUSE_PUBLIC_KEY \
    LANGFUSE_SECRET_KEY; do
    if [[ -n "${!var:-}" ]]; then
        SECRETS_ARGS+=("${var}=${!var}")
    fi
done

# Also load from a .env file in the repo root if present
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "==> Loading secrets from .env"
    while IFS='=' read -r key value; do
        # Skip comments and blank lines
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # Strip inline comments
        value="${value%%#*}"
        # Strip quotes from value
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        # Strip leading/trailing whitespace
        value="${value#"${value%%[![:space:]]*}"}"
        value="${value%"${value##*[![:space:]]}"}"
        SECRETS_ARGS+=("${key}=${value}")
    done < "$REPO_ROOT/.env"
fi

if [[ ${#SECRETS_ARGS[@]} -gt 0 ]]; then
    echo "==> Setting ${#SECRETS_ARGS[@]} secret(s)..."
    fly secrets set --app "$APP_NAME" "${SECRETS_ARGS[@]}"
else
    echo "==> No secrets to push (set env vars or create a .env file)"
fi

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
echo "==> Running fly deploy --remote-only --app ${APP_NAME}"
fly deploy --remote-only --app "$APP_NAME"

echo ""
echo "==> Deploy complete."
echo "    Status:  fly status --app ${APP_NAME}"
echo "    Logs:    fly logs --app ${APP_NAME}"
echo "    Open:    fly open --app ${APP_NAME}"
