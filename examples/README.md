# Examples

## mcp_client.py

An end-to-end example that spawns `atlas-mcp` over stdio and calls three tools:

| Tool called | What it exercises |
|---|---|
| `search_code` | Full classify → retrieve → grade → answer pipeline |
| `find_callers` | Pure call-graph traversal (no LLM) |
| `get_graph_neighborhood` | Bidirectional BFS around a symbol |

### Prerequisites

```bash
cd /path/to/codex-atlas
uv run atlas index src/ --store=memory   # creates data/chunks.json + data/graph.json
```

### Run

```bash
uv run python examples/mcp_client.py          # rich text output
uv run python examples/mcp_client.py --json   # raw JSON (pipe-friendly)
uv run python examples/mcp_client.py --dry-run  # fixture data, no server spawn
```

### Expected output shape

```
[search_code]
  route    : structural
  grade    : 0.75
  attempts : 1
  answer   : '...'
  citations: 1
    - codex_atlas.agent.Agent.run (src/codex_atlas/agent.py:500)

[find_callers]
  target  : codex_atlas.agent.Agent.run
  depth   : 1
  callers : 3
    - codex_atlas.mcp_server.search_code
    ...
```

Exits 0 on success, non-zero if `data/graph.json` is missing or the server fails to start.
