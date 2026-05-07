"""Unit tests for ``Neo4jCallGraph`` and the ``make_call_graph()`` factory.

All tests use a mocked async neo4j driver — no real Neo4j instance required.
The integration path (real AuraDB / Community) is gated behind
``@pytest.mark.integration``.

Mock strategy
-------------
We patch ``AsyncGraphDatabase.driver`` at the module level in
``codex_atlas.indexer.neo4j_graph`` so the constructor never opens a real
socket.  Each test configures the fake session to record Cypher calls or to
return canned ``data()`` results.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    run_side_effect: Any = None,
    data_return: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a fake async session context manager.

    ``run_side_effect``: if set, ``session.run()`` raises this.
    ``data_return``: list of dicts returned by ``result.data()``.
    """
    result_mock = AsyncMock()
    result_mock.data = AsyncMock(return_value=data_return or [])

    session_mock = AsyncMock()
    if run_side_effect is not None:
        session_mock.run = AsyncMock(side_effect=run_side_effect)
    else:
        session_mock.run = AsyncMock(return_value=result_mock)

    # Make session usable as ``async with driver.session() as session``
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    return cm, session_mock, result_mock


def _make_driver(
    run_side_effect: Any = None,
    data_return: list[dict[str, Any]] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (driver_mock, session_mock) configured for a single call pattern."""
    cm, session_mock, _result_mock = _make_session(run_side_effect, data_return)

    driver_mock = MagicMock()
    driver_mock.session = MagicMock(return_value=cm)
    driver_mock.close = AsyncMock()

    return driver_mock, session_mock


# ---------------------------------------------------------------------------
# Env-var fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def neo4j_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the three ATLAS_NEO4J_* env vars that the constructor reads."""
    monkeypatch.setenv("ATLAS_NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("ATLAS_NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("ATLAS_NEO4J_PASSWORD", "test-secret")


# ---------------------------------------------------------------------------
# Import: soft-fail when driver absent
# ---------------------------------------------------------------------------


def test_import_succeeds_without_neo4j_driver() -> None:
    """The module must import cleanly even if 'neo4j' is not installed."""
    # The module is already imported (it's in sys.modules). The point is that
    # importing it did NOT raise ImportError at module level.
    import codex_atlas.indexer.neo4j_graph as m  # noqa: PLC0415

    # _NEO4J_AVAILABLE reflects whether the real package is installed.
    assert isinstance(m._NEO4J_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_constructor_raises_when_uri_missing(neo4j_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError when ATLAS_NEO4J_URI is absent."""
    monkeypatch.delenv("ATLAS_NEO4J_URI")
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock()
        with pytest.raises(RuntimeError, match="ATLAS_NEO4J_URI"):
            Neo4jCallGraph()


def test_constructor_uses_default_username(neo4j_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Username defaults to 'neo4j' when ATLAS_NEO4J_USERNAME is unset."""
    monkeypatch.delenv("ATLAS_NEO4J_USERNAME", raising=False)
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    driver_mock = MagicMock()
    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        Neo4jCallGraph()
        # auth tuple: second element is the (user, password) pair
        call_kwargs = mock_adb.driver.call_args
        auth = call_kwargs[1]["auth"] if "auth" in (call_kwargs[1] or {}) else call_kwargs[0][1]
        assert auth[0] == "neo4j"


# ---------------------------------------------------------------------------
# setup() — DDL applied
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_setup_applies_ddl(neo4j_env: None) -> None:
    """setup() must issue the constraint + index Cypher statements."""
    from codex_atlas.indexer.neo4j_graph import (  # noqa: PLC0415
        _SETUP_CONSTRAINT,
        _SETUP_INDEX,
        Neo4jCallGraph,
    )

    driver_mock, session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    await graph.setup()

    calls = [c.args[0] for c in session_mock.run.call_args_list]
    assert _SETUP_CONSTRAINT in calls, "constraint DDL not issued"
    assert _SETUP_INDEX in calls, "index DDL not issued"


# ---------------------------------------------------------------------------
# add_symbol()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_symbol_issues_merge(neo4j_env: None) -> None:
    """add_symbol() must run a MERGE statement with the right parameters."""
    from codex_atlas.indexer.neo4j_graph import _ADD_SYMBOL, Neo4jCallGraph  # noqa: PLC0415

    driver_mock, session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    await graph.add_symbol("pkg.mod.Foo", "pkg.mod", "class")

    session_mock.run.assert_called_once_with(
        _ADD_SYMBOL, qn="pkg.mod.Foo", m="pkg.mod", k="class"
    )


# ---------------------------------------------------------------------------
# add_edge()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_edge_calls_relationship(neo4j_env: None) -> None:
    """add_edge('calls') must MERGE a :CALLS relationship."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    driver_mock, session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    await graph.add_edge("pkg.a.fn", "pkg.b.fn", "calls")

    cypher = session_mock.run.call_args.args[0]
    assert ":CALLS" in cypher


@pytest.mark.asyncio
async def test_add_edge_imports_relationship(neo4j_env: None) -> None:
    """add_edge('imports') must MERGE an :IMPORTS relationship."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    driver_mock, session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    await graph.add_edge("pkg.a", "pkg.b", "imports")

    cypher = session_mock.run.call_args.args[0]
    assert ":IMPORTS" in cypher


@pytest.mark.asyncio
async def test_add_edge_rejects_unknown_kind(neo4j_env: None) -> None:
    """add_edge() raises ValueError for unknown relationship kinds."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    driver_mock, _session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    with pytest.raises(ValueError, match="Unknown edge kind"):
        await graph.add_edge("a", "b", "defines")


# ---------------------------------------------------------------------------
# find_callers()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_callers_round_trip(neo4j_env: None) -> None:
    """find_callers() returns qualified names from the result records."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    data_return = [
        {"qname": "pkg.a.caller_one"},
        {"qname": "pkg.b.caller_two"},
    ]

    # Build a session whose run() returns a result with our canned data.
    result_mock = AsyncMock()
    result_mock.data = AsyncMock(return_value=data_return)

    session_mock = AsyncMock()
    session_mock.run = AsyncMock(return_value=result_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver_mock = MagicMock()
    driver_mock.session = MagicMock(return_value=cm)

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    callers = await graph.find_callers("pkg.mod.target_fn")
    assert callers == ["pkg.a.caller_one", "pkg.b.caller_two"]


# ---------------------------------------------------------------------------
# find_callees()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_callees_round_trip(neo4j_env: None) -> None:
    """find_callees() returns qualified names from the result records."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    data_return = [{"qname": "pkg.util.helper"}]

    result_mock = AsyncMock()
    result_mock.data = AsyncMock(return_value=data_return)

    session_mock = AsyncMock()
    session_mock.run = AsyncMock(return_value=result_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver_mock = MagicMock()
    driver_mock.session = MagicMock(return_value=cm)

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    callees = await graph.find_callees("pkg.mod.caller_fn")
    assert callees == ["pkg.util.helper"]


# ---------------------------------------------------------------------------
# import_chain()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_chain_multi_hop(neo4j_env: None) -> None:
    """import_chain() returns path objects for multi-hop traversals."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    # Simulate two paths returned (one 1-hop, one 2-hop).
    path_a = MagicMock(name="path_a")
    path_b = MagicMock(name="path_b")
    data_return = [{"path": path_a}, {"path": path_b}]

    result_mock = AsyncMock()
    result_mock.data = AsyncMock(return_value=data_return)

    session_mock = AsyncMock()
    session_mock.run = AsyncMock(return_value=result_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver_mock = MagicMock()
    driver_mock.session = MagicMock(return_value=cm)

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    paths = await graph.import_chain("pkg.mod.Symbol")
    assert paths == [path_a, path_b]

    # Verify the Cypher contains the IMPORTS* pattern.
    cypher = session_mock.run.call_args.args[0]
    assert "IMPORTS*1..3" in cypher


# ---------------------------------------------------------------------------
# get_neighborhood()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_neighborhood_uses_depth(neo4j_env: None) -> None:
    """get_neighborhood() embeds the depth into the Cypher query."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    data_return = [{"qname": "pkg.x.Node"}]

    result_mock = AsyncMock()
    result_mock.data = AsyncMock(return_value=data_return)

    session_mock = AsyncMock()
    session_mock.run = AsyncMock(return_value=result_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session_mock)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver_mock = MagicMock()
    driver_mock.session = MagicMock(return_value=cm)

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    nodes = await graph.get_neighborhood("pkg.mod.Root", depth=2)
    assert nodes == ["pkg.x.Node"]

    cypher = session_mock.run.call_args.args[0]
    assert "[*1..2]" in cypher


@pytest.mark.asyncio
async def test_get_neighborhood_rejects_zero_depth(neo4j_env: None) -> None:
    """get_neighborhood() raises ValueError for depth < 1."""
    from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

    driver_mock = MagicMock()
    driver_mock.session = MagicMock()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    with pytest.raises(ValueError, match="depth must be >= 1"):
        await graph.get_neighborhood("pkg.mod.X", depth=0)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_issues_detach_delete(neo4j_env: None) -> None:
    """clear() must issue a DETACH DELETE statement."""
    from codex_atlas.indexer.neo4j_graph import _CLEAR, Neo4jCallGraph  # noqa: PLC0415

    driver_mock, session_mock = _make_driver()

    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        graph = Neo4jCallGraph()

    await graph.clear()

    session_mock.run.assert_called_once_with(_CLEAR)


# ---------------------------------------------------------------------------
# make_call_graph() factory — env-var dispatch
# ---------------------------------------------------------------------------


def test_make_call_graph_default_returns_networkx(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_call_graph() with no env var returns a NetworkX CallGraph."""
    monkeypatch.delenv("ATLAS_GRAPH_BACKEND", raising=False)
    from codex_atlas.indexer import make_call_graph  # noqa: PLC0415
    from codex_atlas.indexer.graph import CallGraph  # noqa: PLC0415

    graph = make_call_graph()
    assert isinstance(graph, CallGraph)


def test_make_call_graph_networkx_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_call_graph() with ATLAS_GRAPH_BACKEND=networkx returns CallGraph."""
    monkeypatch.setenv("ATLAS_GRAPH_BACKEND", "networkx")
    from codex_atlas.indexer import make_call_graph  # noqa: PLC0415
    from codex_atlas.indexer.graph import CallGraph  # noqa: PLC0415

    graph = make_call_graph()
    assert isinstance(graph, CallGraph)


def test_make_call_graph_neo4j_dispatches(
    neo4j_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """make_call_graph() with ATLAS_GRAPH_BACKEND=neo4j returns Neo4jCallGraph."""
    monkeypatch.setenv("ATLAS_GRAPH_BACKEND", "neo4j")

    driver_mock = MagicMock()
    with patch(
        "codex_atlas.indexer.neo4j_graph.AsyncGraphDatabase"
    ) as mock_adb, patch(
        "codex_atlas.indexer.neo4j_graph._NEO4J_AVAILABLE", True
    ):
        mock_adb.driver = MagicMock(return_value=driver_mock)
        from codex_atlas.indexer import make_call_graph  # noqa: PLC0415
        from codex_atlas.indexer.neo4j_graph import Neo4jCallGraph  # noqa: PLC0415

        graph = make_call_graph()
        assert isinstance(graph, Neo4jCallGraph)


def test_make_call_graph_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """make_call_graph() raises RuntimeError for unknown backend names."""
    monkeypatch.setenv("ATLAS_GRAPH_BACKEND", "faiss")
    from codex_atlas.indexer import make_call_graph  # noqa: PLC0415

    with pytest.raises(RuntimeError, match="Unknown ATLAS_GRAPH_BACKEND"):
        make_call_graph()
