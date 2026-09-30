"""The published client sends the API contract without model dependencies."""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import pytest

from src.truetype import TrueTypeClient, TrueTypeError
from src.truetype import client as client_module


def test_client_sends_auth_and_returns_answers(monkeypatch):
    seen = {}

    def fake_open(request, timeout):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.header_items())
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return io.BytesIO(b'{"model":"gemma-4-12b","answers":{"q":{"type":"noul","noul":0.9}},"usage":{}}')

    monkeypatch.setattr(client_module, "urlopen", fake_open)
    questions = {"q": {"type": "noul", "instructions": "Is this urgent?"}}
    result = TrueTypeClient("https://example.hf.space/", token="secret", timeout=5).system_one(
        state={"text": "urgent"}, questions=questions
    )

    assert result["answers"]["q"]["noul"] == 0.9
    assert seen["url"] == "https://example.hf.space/v1/systemone"
    assert seen["method"] == "POST"
    assert seen["headers"]["Authorization"] == "Bearer secret"
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["payload"]["state"] == {"text": "urgent"}
    assert seen["payload"]["questions"] == questions
    assert seen["timeout"] == 5


def test_client_surfaces_api_validation_error(monkeypatch):
    def fake_open(request, timeout):
        raise HTTPError(request.full_url, 422, "Unprocessable Entity", {},
                        io.BytesIO(b'{"detail":"bad question"}'))

    monkeypatch.setattr(client_module, "urlopen", fake_open)
    with pytest.raises(TrueTypeError, match="HTTP 422: bad question"):
        TrueTypeClient("https://example.hf.space").system_one(
            state="hello", questions={"q": {"type": "noul", "instructions": "?"}}
        )


def test_client_rejects_invalid_setup():
    with pytest.raises(ValueError, match="endpoint"):
        TrueTypeClient("example.hf.space")
    with pytest.raises(ValueError, match="questions"):
        TrueTypeClient("https://example.hf.space").system_one(state="hello", questions={})
