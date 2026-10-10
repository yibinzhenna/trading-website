"""
Request-size limit: a huge body is refused before it is read.
"""

import time
import tracemalloc

from fastapi.testclient import TestClient

from api import deps

OK = {"symbol": "SPY", "kind": "momentum"}


def client(tmp_path, **overrides):
    settings = dict(provider="local", data_root="tests/fixtures",
                    cache_root=str(tmp_path / "cache"), rate_limit=10_000,
                    max_inflight=1_000, max_body_bytes=64 * 1024)
    settings.update(overrides)
    deps.reset_for_tests(**settings)
    from api.main import app
    return TestClient(app)


def huge_json(mb):
    n = int(mb * 1e6 / 10)
    return '{"symbol":"SPY","kind":"momentum","params":{' + ",".join(
        f'"k{i}":1' for i in range(n)) + "}}"


def test_normal_requests_pass(tmp_path):
    with client(tmp_path) as c:
        assert c.post("/backtest", json=OK).status_code == 202


def test_declared_oversize_is_refused_without_reading(tmp_path):
    """The live hazard: 18 MB took 19 s and 500 MB of memory to reject."""
    body = huge_json(18)
    with client(tmp_path) as c:
        tracemalloc.start()
        before = tracemalloc.get_traced_memory()[0]
        t = time.monotonic()
        r = c.post("/backtest", content=body,
                   headers={"content-type": "application/json"})
        elapsed = time.monotonic() - t
        peak = tracemalloc.get_traced_memory()[1] - before
        tracemalloc.stop()
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    assert elapsed < 2.0
    assert peak < 60e6, f"peak {peak / 1e6:.0f} MB"   # the body itself is 18 MB


def test_undeclared_size_is_counted_as_it_arrives(tmp_path):
    """Chunked upload, no Content-Length: still cut off at the limit."""
    def chunks():
        for _ in range(64):
            yield b"x" * 4096          # 256 KB in total
    with client(tmp_path) as c:
        r = c.post("/backtest", content=chunks(),
                   headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_counting_applies_even_when_a_length_is_declared(tmp_path):
    """Declared small, sent large: counting still catches it. (A real HTTP
    server frames the body by Content-Length, so this is belt and braces.)"""
    import asyncio
    from api.limits import BodySizeLimit

    sent, statuses = [b"y" * 70_000], []

    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"read it all"})

    async def receive():
        return {"type": "http.request", "body": sent.pop() if sent else b"",
                "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    deps.settings.max_body_bytes = 64 * 1024
    scope = {"type": "http", "headers": [(b"content-length", b"10")]}
    asyncio.run(BodySizeLimit(app)(scope, receive, send))
    assert statuses == [413]


def test_garbage_content_length_is_a_400(tmp_path):
    with client(tmp_path) as c:
        r = c.post("/backtest", content=b"{}",
                   headers={"content-type": "application/json",
                            "content-length": "lots"})
    assert r.status_code == 400


def test_limit_is_configurable(tmp_path):
    with client(tmp_path, max_body_bytes=10) as c:
        assert c.post("/backtest", json=OK).status_code == 413
