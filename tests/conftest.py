"""Pytest configuration shared across the suite.

Why a fixture for event-loop cleanup:

  * Several CLI tests use ``typer.testing.CliRunner`` to invoke commands
    that internally call ``asyncio.run(...)``. After each such call the
    thread-local event loop is left as ``None`` (correct for production).
  * pytest-asyncio's per-test teardown — ``_temporary_event_loop_policy``
    in ``plugin.py`` — captures the *current* event loop via the
    deprecated ``asyncio.get_event_loop()`` API. When there is no loop
    on the thread, that call AUTO-CREATES a new ``_UnixSelectorEventLoop``
    (with its self-pipe socket pair) and never closes it.
  * The result is a string of ``ResourceWarning: unclosed
    <socket.socket fd=12, ...>`` plus ``unclosed event loop ...``
    warnings that pytest's unraisable-exception hook eventually escalates
    to a test failure.

The fix is small: after each test, if a loop has been auto-created and
left open on this thread, close it and clear the thread-local reference
so pytest-asyncio's next teardown captures ``None`` again instead of
inheriting a soon-to-be-leaked loop.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _close_leaked_event_loop() -> Iterator[None]:
    """Close any thread-local event loop left behind by pytest-asyncio.

    Runs after every test in the suite (sync or async). When the test
    leaves a non-running, non-closed loop attached to the current
    thread, we close it and detach the reference so the next test's
    pytest-asyncio teardown does not inherit it.
    """
    yield
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            # No loop attached — pytest-asyncio's teardown won't touch
            # the policy in a way that auto-creates one.
            return
    if loop.is_running() or loop.is_closed():
        return
    loop.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.get_event_loop_policy().set_event_loop(None)
