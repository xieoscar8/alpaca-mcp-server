"""Shared fixtures for the Alpaca MCP Server test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def canonical_paper_configuration(monkeypatch, request):
    """Local construction defaults; negative tests explicitly override.

    No automatic broker cleanup, even with credentials or integration opt-in.
    V2 has no bulk cleanup tools; cleanup requires separate operator approval.
    """
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("ALPACA_PAPER_TRADE", "true")
