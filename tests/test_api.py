import json

import pytest
from fastapi.testclient import TestClient

import main
from ask_service import AskError

H = {"X-App-Token": "test-token"}
BODY = {
    "question": "total?",
    "today": "2026-11-15",
    "expenses": [{"title": "Tea", "amount": 10, "category": "Food", "date": "2026-11-01"}],
}


@pytest.fixture(autouse=True)
def reset_limits():
    main._hits.clear()


client = TestClient(main.app)


def test_health_open():
    assert client.get("/health").status_code == 200


def test_requires_token():
    assert client.post("/ask", json=BODY).status_code == 401


def test_bad_date_rejected():
    bad = {**BODY, "today": "15/11/2026"}
    assert client.post("/ask", json=bad, headers=H).status_code == 422


def test_empty_question():
    assert client.post("/ask", json={**BODY, "question": " "}, headers=H).status_code == 400


def test_ask_ok(monkeypatch):
    monkeypatch.setattr(main, "run_ask", lambda *a, **k: iter(["Hello ", "there"]))
    r = client.post("/ask", json=BODY, headers=H)
    assert r.json() == {"answer": "Hello there"}


def test_stream_ndjson(monkeypatch):
    monkeypatch.setattr(main, "run_ask", lambda *a, **k: iter(["a", "b"]))
    r = client.post("/ask/stream", json=BODY, headers=H)
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert lines == [{"delta": "a"}, {"delta": "b"}, {"done": True}]


def test_stream_quota_is_http_429(monkeypatch):
    def boom(*a, **k):
        raise AskError(429, "AI is busy right now, try again in a minute")
        yield

    monkeypatch.setattr(main, "run_ask", boom)
    r = client.post("/ask/stream", json=BODY, headers=H)
    assert r.status_code == 429


def test_stream_midway_error_line(monkeypatch):
    def gen(*a, **k):
        yield "a"
        raise AskError(502, "AI service failed")

    monkeypatch.setattr(main, "run_ask", gen)
    lines = [json.loads(x) for x in client.post("/ask/stream", json=BODY, headers=H).text.splitlines()]
    assert lines == [{"delta": "a"}, {"error": "AI service failed"}]
