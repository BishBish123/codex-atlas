# ADR-004: MCP server tool shape

## Status

Accepted.

## Context

FastMCP exposes Python functions as MCP tools and Pydantic models as
their input/output schemas. The decision is which tools to surface and
how to scope them.

Two failure modes to design around:

- **Too few tools.** Everything funnels through `search_code`; the
  calling LLM always gets a synthesised answer when it sometimes wants
  raw graph results.
- **Too many tools.** Every internal function becomes a tool; the
  calling LLM has to read 30 docstrings to pick the right one.

## Decision

Seven tools and one resource:

| name | purpose | route |
| --- | --- | --- |
| `search_code(query, top_k)` | adaptive | router decides |
| `explain_function(qualified_name)` | structural | forced |
| `find_callers(qualified_name, depth)` | graph-only | graph |
| `summarize_module(module_path)` | summarisation | forced |
| `search_codebase(query, top_k, route?)` | adaptive + override | router or forced |
| `get_graph_neighborhood(symbol, depth)` | graph-only | graph |
| `explain(symbol)` | adaptive | router |
| `codebase://stats` | resource | none |

## Rationale

### Why these names

The four "verbal" tools (`search_code`, `explain_function`,
`find_callers`, `summarize_module`) match phrases a calling LLM
naturally produces. The two "graph" tools (`find_callers`,
`get_graph_neighborhood`) are pure graph ops with no LLM involved —
they answer queries deterministically and cheaply.

`search_codebase` exposes the explicit-route override for cases where
the calling LLM has *already* decided which pipeline it wants
(e.g. it parsed the query itself and concluded "structural"). This
saves a round-trip through the classifier.

`explain(symbol)` is the everything-tool: hand it a qualified name and
it runs the full agent loop. Costlier than `find_callers`, but the
right call when the LLM wants prose instead of names.

### Why a resource for stats

`codebase://stats` describes the indexed corpus (node/edge counts,
language). MCP resources are pull-only and side-effect-free — exactly
right for a `cat`-style observation that doesn't change state.

### Why not 30 tools

We resisted the temptation to expose every internal function. The
heuristic was: *can the calling LLM use this tool without reading
hundreds of lines of source?* If not, fold it behind one of the seven
above. Examples we *didn't* expose:

- `classify(query)` — internal step; calling LLMs that want to inspect
  the routing decision get it via `search_code`'s response.
- `hybrid_score(...)` — only meaningful inside the retriever.
- `parse_corpus(...)` — index-time, not query-time.

## Consequences

- Adding a tool requires adding a Pydantic response model and a
  one-line FastMCP decorator. The bar to add is intentionally low,
  but each one is named after what the calling LLM is *trying to do*,
  not what's convenient internally.
- The schema descriptions in the Pydantic models are visible to the
  calling LLM. Each field has a docstring or a `Field(description=...)`
  so the schema reads like a tool manual.
- All tools raise `ValueError` on bad input rather than returning an
  error model. FastMCP translates this into the standard MCP error
  envelope, which calling LLMs handle natively.

## Alternatives considered

- **One mega-tool with a `mode` parameter.** Loses the schema
  surface — every input/output field becomes optional, and the
  calling LLM can't reason about which fields apply.
- **Streaming responses.** FastMCP supports it; we don't use it
  because synthesised answers are short (<1KB) and graph results are
  bounded by `top_k`.
