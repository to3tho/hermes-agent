"""
Tests for the /api/usage/summary read-only aggregate endpoint.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _read_usage_summary,
    cors_middleware,
)
from hermes_state import SessionDB


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra: Dict[str, Any] = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/usage/summary", adapter._handle_usage_summary)
    return app


def _seed_sessions(db_path):
    """Create a few sessions with realistic token counts."""
    db = SessionDB(db_path)
    try:
        db.create_session("s1", "cli", model="claude-opus-4.6")
        db.update_token_counts(
            "s1",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_write_tokens=50,
            estimated_cost_usd=0.05,
        )
        db.create_session("s2", "cli", model="claude-opus-4.6")
        db.update_token_counts(
            "s2",
            input_tokens=2000,
            output_tokens=800,
        )
        db.create_session("s3", "api_server", model="gpt-5.5")
        db.update_token_counts(
            "s3",
            input_tokens=500,
            output_tokens=200,
            estimated_cost_usd=0.02,
        )
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Module-level — _read_usage_summary
# ---------------------------------------------------------------------------


class TestReadUsageSummary:
    def test_empty_db_returns_zero_totals(self, tmp_path):
        db_path = tmp_path / "state.db"
        SessionDB(db_path).close()  # create empty DB
        out = _read_usage_summary(db_path)
        assert out["tokens"] == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        assert out["estimated_cost_usd"] is None
        assert out["by_model"] == []

    def test_aggregates_token_totals(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_usage_summary(db_path)
        # s1 + s2 + s3
        assert out["tokens"]["input"] == 1000 + 2000 + 500
        assert out["tokens"]["output"] == 500 + 800 + 200
        assert out["tokens"]["cache_read"] == 200
        assert out["tokens"]["cache_write"] == 50

    def test_aggregates_cost_when_present(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_usage_summary(db_path)
        # s1 (0.05) + s3 (0.02) = 0.07; s2 had no cost.
        assert out["estimated_cost_usd"] == pytest.approx(0.07)

    def test_by_model_breakdown_ordered_by_tokens(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        out = _read_usage_summary(db_path)
        models = {row["model_id"]: row for row in out["by_model"]}
        assert "claude-opus-4.6" in models
        assert "gpt-5.5" in models
        # claude has more total tokens (s1+s2 = 4300) vs gpt-5.5 (700)
        ordered = [row["model_id"] for row in out["by_model"]]
        assert ordered.index("claude-opus-4.6") < ordered.index("gpt-5.5")

    def test_since_window_filters_older_sessions(self, tmp_path):
        db_path = tmp_path / "state.db"
        _seed_sessions(db_path)
        # Use a far-future timestamp; should match nothing.
        out = _read_usage_summary(db_path, since_iso="9999-01-01T00:00:00Z")
        assert out["tokens"] == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
        }

    def test_missing_columns_yield_empty_summary_not_error(self, tmp_path):
        """If a legacy DB is missing one of the cache_* columns, the
        endpoint must degrade gracefully to zeros rather than 500."""
        db_path = tmp_path / "legacy.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE sessions (id TEXT, model TEXT, started_at TEXT)"
            )
        out = _read_usage_summary(db_path)
        assert out["tokens"] == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        assert out["estimated_cost_usd"] is None


# ---------------------------------------------------------------------------
# HTTP integration
# ---------------------------------------------------------------------------


class TestUsageSummaryEndpoint:
    @pytest.mark.asyncio
    async def test_returns_zero_when_db_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/usage/summary")
            assert resp.status == 200
            body = await resp.json()
        assert body["tokens"] == {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
        }
        assert body["by_model"] == []

    @pytest.mark.asyncio
    async def test_returns_aggregated_data_when_db_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db")
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/usage/summary")
            body = await resp.json()
        assert body["tokens"]["input"] == 3500
        assert body["tokens"]["output"] == 1500
        assert len(body["by_model"]) == 2

    @pytest.mark.asyncio
    async def test_requires_auth_when_key_configured(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/usage/summary")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_since_param_propagates_to_helper(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_sessions(tmp_path / "state.db")
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(
                "/api/usage/summary",
                params={"since": "9999-01-01T00:00:00Z"},
            )
            body = await resp.json()
        # Future timestamp filters out everything.
        assert body["tokens"]["input"] == 0
