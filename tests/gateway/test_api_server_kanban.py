"""
Tests for the light kanban CRUD surface on the API server.

Endpoints under test:

  * ``GET   /api/kanban/boards``
  * ``POST  /api/kanban/boards``
  * ``DELETE /api/kanban/boards/{slug}``
  * ``GET   /api/kanban/tasks``
  * ``POST  /api/kanban/tasks``
  * ``DELETE /api/kanban/tasks/{task_id}``
  * ``POST  /api/kanban/tasks/{task_id}/complete``

Each test isolates the kanban home via ``HERMES_KANBAN_HOME`` so
the suite never touches the user's real ``~/.hermes/kanban``.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra: Dict[str, Any] = {"key": api_key} if api_key else {}
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/kanban/boards", adapter._handle_kanban_list_boards)
    app.router.add_post("/api/kanban/boards", adapter._handle_kanban_create_board)
    app.router.add_delete(
        "/api/kanban/boards/{slug}", adapter._handle_kanban_delete_board
    )
    app.router.add_get("/api/kanban/tasks", adapter._handle_kanban_list_tasks)
    app.router.add_post("/api/kanban/tasks", adapter._handle_kanban_create_task)
    app.router.add_delete(
        "/api/kanban/tasks/{task_id}", adapter._handle_kanban_archive_task
    )
    app.router.add_post(
        "/api/kanban/tasks/{task_id}/complete",
        adapter._handle_kanban_complete_task,
    )
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    return app


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Pin the kanban root to a per-test tmpdir.

    The kanban subsystem walks ``HERMES_KANBAN_HOME`` first, so this
    fixture is the single source of truth. We also wipe the
    connection cache on the way out so test order can't leak state."""
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Force a fresh connection on the first call by clearing the
    # module-level path cache that kanban_db keeps.
    try:
        from hermes_cli import kanban_db
        kanban_db._INITIALIZED_PATHS.clear()
    except (ImportError, AttributeError):
        pass
    return tmp_path


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------


class TestKanbanBoards:
    @pytest.mark.asyncio
    async def test_list_boards_includes_default(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/kanban/boards")
            assert resp.status == 200
            body = await resp.json()
        slugs = [b.get("slug") for b in body["boards"]]
        assert "default" in slugs

    @pytest.mark.asyncio
    async def test_create_board_then_appears_in_list(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/boards",
                json={"slug": "marketing", "name": "Marketing"},
            )
            assert resp.status == 200
            created = await resp.json()
            assert created["board"]["slug"] == "marketing"

            resp = await cli.get("/api/kanban/boards")
            body = await resp.json()
        slugs = [b.get("slug") for b in body["boards"]]
        assert "marketing" in slugs

    @pytest.mark.asyncio
    async def test_create_board_requires_slug(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/kanban/boards", json={})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_delete_board_archives_by_default(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            await cli.post(
                "/api/kanban/boards", json={"slug": "throwaway"}
            )
            resp = await cli.delete("/api/kanban/boards/throwaway")
            assert resp.status == 200
            body = await resp.json()
        assert body["action"] == "archived"
        assert body["slug"] == "throwaway"

    @pytest.mark.asyncio
    async def test_delete_default_board_is_rejected(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.delete("/api/kanban/boards/default")
            assert resp.status == 400


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestKanbanTasks:
    @pytest.mark.asyncio
    async def test_create_task_returns_serialised_task(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/tasks",
                json={"title": "Plan PR-3 follow-up", "priority": 2},
            )
            assert resp.status == 200
            body = await resp.json()
        task = body["task"]
        assert task["title"] == "Plan PR-3 follow-up"
        assert task["priority"] == 2
        assert task["status"] in ("ready", "todo")
        assert task["id"].startswith("t_")

    @pytest.mark.asyncio
    async def test_create_task_requires_title(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/api/kanban/tasks", json={"title": ""})
            assert resp.status == 400

    @pytest.mark.asyncio
    async def test_list_tasks_returns_created_task(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            await cli.post("/api/kanban/tasks", json={"title": "alpha"})
            await cli.post("/api/kanban/tasks", json={"title": "beta"})
            resp = await cli.get("/api/kanban/tasks")
            assert resp.status == 200
            body = await resp.json()
        titles = sorted(t["title"] for t in body["tasks"])
        assert titles == ["alpha", "beta"]

    @pytest.mark.asyncio
    async def test_archive_task_then_excluded_from_list(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/tasks", json={"title": "to be archived"}
            )
            task_id = (await resp.json())["task"]["id"]

            resp = await cli.delete(f"/api/kanban/tasks/{task_id}")
            assert resp.status == 200

            resp = await cli.get("/api/kanban/tasks")
            body = await resp.json()
        assert all(t["id"] != task_id for t in body["tasks"])

    @pytest.mark.asyncio
    async def test_complete_task_transitions_to_done(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/tasks", json={"title": "to be completed"}
            )
            task_id = (await resp.json())["task"]["id"]

            resp = await cli.post(
                f"/api/kanban/tasks/{task_id}/complete",
                json={"result": "all good"},
            )
            assert resp.status == 200
            body = await resp.json()
        assert body["task"]["status"] == "done"
        assert body["task"]["result"] == "all good"

    @pytest.mark.asyncio
    async def test_complete_nonexistent_task_is_404(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/api/kanban/tasks/t_doesnotexist00/complete", json={}
            )
            assert resp.status == 404

    @pytest.mark.asyncio
    async def test_idempotency_key_header_dedupes_create(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            r1 = await cli.post(
                "/api/kanban/tasks",
                json={"title": "idempo"},
                headers={"Idempotency-Key": "abc123"},
            )
            r2 = await cli.post(
                "/api/kanban/tasks",
                json={"title": "idempo"},
                headers={"Idempotency-Key": "abc123"},
            )
            t1 = (await r1.json())["task"]["id"]
            t2 = (await r2.json())["task"]["id"]
        assert t1 == t2


# ---------------------------------------------------------------------------
# Auth + capabilities
# ---------------------------------------------------------------------------


class TestKanbanAuth:
    @pytest.mark.asyncio
    async def test_endpoints_require_auth_when_key_configured(
        self, kanban_home
    ):
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/kanban/boards")
            assert resp.status == 401
            resp = await cli.get(
                "/api/kanban/boards",
                headers={"Authorization": "Bearer sk-secret"},
            )
            assert resp.status == 200


class TestCapabilitiesAdvertisesKanban:
    @pytest.mark.asyncio
    async def test_capabilities_lists_kanban_endpoints(self, kanban_home):
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/capabilities")
            body = await resp.json()
        assert body["features"]["remote_kanban"] is True
        endpoints = body["endpoints"]
        assert endpoints["kanban_boards"]["path"] == "/api/kanban/boards"
        assert endpoints["kanban_tasks"]["path"] == "/api/kanban/tasks"
        assert (
            endpoints["kanban_task_complete"]["path"]
            == "/api/kanban/tasks/{task_id}/complete"
        )
