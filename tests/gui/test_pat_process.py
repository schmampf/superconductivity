from __future__ import annotations

import socket
import urllib.error

import pytest

pytest.importorskip("panel")

from superconductivity.optimizers.pat.process import (
    _assert_local_socket_bindable,
    _reserve_local_port,
    _wait_for_server_ready,
)


class _Response:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_wait_for_server_ready_succeeds(monkeypatch) -> None:
    attempts = {"count": 0}

    def fake_urlopen(url: str, timeout: float = 1.0) -> _Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise urllib.error.URLError("not ready")
        return _Response(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert _wait_for_server_ready(
        "http://127.0.0.1:12345/",
        timeout=0.5,
        interval=0.01,
    )


def test_assert_local_socket_bindable_raises_clear_error(monkeypatch) -> None:
    original_bind = socket.socket.bind

    def fake_bind(self: socket.socket, address: tuple[str, int]) -> None:
        raise PermissionError("sandboxed")

    monkeypatch.setattr(socket.socket, "bind", fake_bind)

    with pytest.raises(RuntimeError, match="Cannot bind a local TCP socket"):
        _assert_local_socket_bindable()

    monkeypatch.setattr(socket.socket, "bind", original_bind)


def test_reserve_local_port_returns_bound_port(monkeypatch) -> None:
    class _FakeSocket:
        def __enter__(self) -> "_FakeSocket":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def bind(self, address: tuple[str, int]) -> None:
            self.bound = address

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 54321)

    monkeypatch.setattr("socket.socket", lambda *args, **kwargs: _FakeSocket())

    assert _reserve_local_port() == 54321
