"""Single, fail-closed Paper configuration and transport boundary."""

import os

import httpx

PAPER_URL = "https://paper-api.alpaca.markets"


def paper_enabled() -> bool:
    return os.environ.get("ALPACA_PAPER_TRADE") == "true"


def require_paper() -> str:
    if not paper_enabled():
        raise ValueError("ALPACA_PAPER_TRADE must be exactly true")
    return PAPER_URL


def paper_client(client: object) -> bool:
    """Reject unknown targets, redirects, alternate ports, paths and credentials."""
    return (
        paper_enabled()
        and getattr(client, "base_url", None) in (httpx.URL(PAPER_URL), httpx.URL(PAPER_URL + "/"))
        and getattr(client, "follow_redirects", None) is False
    )
