"""
Tests for `?profile=<name>` scoping on /api/tools/toolsets.

PR `feat/tools-profile-scoped` (plan v11): the GET (list) and
PUT (toggle) handlers learn to read the targeted profile's
``config.yaml`` instead of the HERMES_HOME-pinned default.
This avoids the previous race where a CLI ``hermes profile
use mira-uitest`` had no effect on the running gateway process.

Test surface:

* GET reads the named profile's config (returns its enabled set).
* PUT writes back into the named profile's config.yaml.
* Back-compat: no `?profile=` → reads/writes the process default,
  byte-identical to the pre-PR behaviour.
* Invalid profile → 400 (matches `_resolve_profile_dir`).
* Cross-profile isolation: PUT on profile A leaves profile B's
  config.yaml untouched (hash check).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
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
    app.router.add_get("/api/tools/toolsets", adapter._handle_list_toolsets)
    app.router.add_put("/api/tools/toolsets/{key}", adapter._handle_set_toolset)
    return app


def _seed_profile(hermes_home: Path, name: str, enabled_toolsets: list[str]) -> Path:
    """Create a profile directory + minimal config.yaml with the given
    enabled toolsets pinned for the api_server platform.

    The 'default' profile uses ~/.hermes/ directly; named profiles
    use ~/.hermes/profiles/<name>/.
    """
    if name == "default":
        profile_dir = hermes_home
    else:
        profile_dir = hermes_home / "profiles" / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "platform_toolsets": {"api_server": enabled_toolsets},
    }
    (profile_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return profile_dir


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class TestListToolsetsByProfile:
    @pytest.mark.asyncio
    async def test_no_profile_reads_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", ["web", "terminal"])
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/tools/toolsets")
            assert resp.status == 200
            body = await resp.json()
        # The default profile's enabled set should win.
        enabled_keys = {t["key"] for t in body["toolsets"] if t["enabled"]}
        assert "web" in enabled_keys
        assert "terminal" in enabled_keys
        assert body.get("profile") is None

    @pytest.mark.asyncio
    async def test_named_profile_reads_its_own_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Default profile has different toolsets so we can tell them apart.
        _seed_profile(tmp_path, "default", ["web"])
        _seed_profile(tmp_path, "mira-uitest", ["browser", "file"])
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/tools/toolsets?profile=mira-uitest")
            assert resp.status == 200
            body = await resp.json()
        enabled_keys = {t["key"] for t in body["toolsets"] if t["enabled"]}
        # mira-uitest's set, NOT default's.
        assert "browser" in enabled_keys
        assert "file" in enabled_keys
        assert "web" not in enabled_keys
        assert body["profile"] == "mira-uitest"

    @pytest.mark.asyncio
    async def test_invalid_profile_returns_400(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", [])
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/tools/toolsets?profile=does-not-exist")
            assert resp.status == 400
            body = await resp.json()
        assert "error" in body


class TestSetToolsetByProfile:
    @pytest.mark.asyncio
    async def test_named_profile_write_lands_in_correct_yaml(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", ["web"])
        mira_dir = _seed_profile(tmp_path, "mira-uitest", [])

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/tools/toolsets/browser?profile=mira-uitest",
                json={"enabled": True},
            )
            assert resp.status == 200
            body = await resp.json()
        assert body["ok"] is True
        assert body["enabled"] is True
        assert body["profile"] == "mira-uitest"

        # Verify the WRITE landed in mira-uitest, not in default.
        mira_cfg = yaml.safe_load(
            (mira_dir / "config.yaml").read_text(encoding="utf-8")
        )
        assert "browser" in mira_cfg.get("platform_toolsets", {}).get(
            "api_server", []
        )

    @pytest.mark.asyncio
    async def test_named_profile_write_does_NOT_touch_default(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", ["web"])
        _seed_profile(tmp_path, "mira-uitest", [])
        default_cfg_path = tmp_path / "config.yaml"
        default_sha_before = _sha256_file(default_cfg_path)

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/tools/toolsets/browser?profile=mira-uitest",
                json={"enabled": True},
            )
            assert resp.status == 200

        default_sha_after = _sha256_file(default_cfg_path)
        assert default_sha_before == default_sha_after, (
            "PUT on mira-uitest profile must not modify the default "
            "profile's config.yaml — byte-identical sha256 required"
        )

    @pytest.mark.asyncio
    async def test_no_profile_writes_to_default(self, tmp_path, monkeypatch):
        """Back-compat: omitting ?profile= mutates the default
        (HERMES_HOME) config exactly as before."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", [])

        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/tools/toolsets/web",
                json={"enabled": True},
            )
            assert resp.status == 200
            body = await resp.json()
        assert body["profile"] is None

        default_cfg = yaml.safe_load(
            (tmp_path / "config.yaml").read_text(encoding="utf-8")
        )
        assert "web" in default_cfg.get("platform_toolsets", {}).get(
            "api_server", []
        )

    @pytest.mark.asyncio
    async def test_invalid_profile_on_PUT_returns_400(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", [])
        app = _create_app(_make_adapter())
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/tools/toolsets/browser?profile=does-not-exist",
                json={"enabled": True},
            )
            assert resp.status == 400


class TestAuthStillRequired:
    @pytest.mark.asyncio
    async def test_GET_requires_auth_when_key_configured(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", [])
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/tools/toolsets?profile=default")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_PUT_requires_auth_when_key_configured(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _seed_profile(tmp_path, "default", [])
        app = _create_app(_make_adapter(api_key="sk-secret"))
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.put(
                "/api/tools/toolsets/web?profile=default",
                json={"enabled": True},
            )
            assert resp.status == 401
