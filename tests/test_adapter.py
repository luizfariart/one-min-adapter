"""Unit tests for the 1min.ai adapter — mapping, prompt flattening, error paths.

Pure logic only: no network, no real 1min.ai calls. The translation functions
are the behaviour contract; the HTTP layer is a thin wrapper over them.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "one_min_adapter",
    Path(__file__).resolve().parent.parent / "one_min_adapter.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


class TestModelMapping:
    def test_known_model_maps(self):
        assert _mod._map_model("deepseek-v4-flash") == "deepseek-flash"
        assert _mod._map_model("gpt-4o-mini") == "gpt-4o-mini"
        assert _mod._map_model("claude-4.6-sonnet") == "us.anthropic.claude-sonnet-4-6"

    def test_unknown_model_passes_through(self):
        assert _mod._map_model("some/future-model") == "some/future-model"

    def test_empty_model_uses_default(self):
        assert _mod._map_model(None) == "deepseek-flash"
        assert _mod._map_model("") == "deepseek-flash"


class TestBuildPrompt:
    def test_system_and_user_flatten(self):
        out = _mod._build_prompt([
            {"role": "system", "content": "responda curto"},
            {"role": "user", "content": "qual a capital do Brasil?"},
        ])
        assert "System instructions:" in out
        assert "responda curto" in out
        assert "qual a capital do Brasil?" in out

    def test_assistant_turns_are_framed(self):
        out = _mod._build_prompt([
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "olá"},
            {"role": "user", "content": "tudo bem?"},
        ])
        assert "Assistant: olá" in out

    def test_no_system_no_prefix(self):
        out = _mod._build_prompt([{"role": "user", "content": "oi"}])
        assert "System instructions:" not in out
        assert out == "oi"

    def test_empty_messages(self):
        assert _mod._build_prompt([]) == ""


class TestExtractText:
    def test_result_object_list(self):
        record = {"aiRecordDetail": {"resultObject": ["Bras", "ília"]}}
        assert _mod._extract_text(record) == "Brasília"

    def test_result_object_string(self):
        record = {"aiRecordDetail": {"resultObject": "resposta"}}
        assert _mod._extract_text(record) == "resposta"

    def test_missing_detail(self):
        assert _mod._extract_text({}) == ""


class TestErrorEnvelope:
    def test_openai_error_shape(self):
        resp = _mod._openai_error(429, "rate_limit_exceeded", "credits exhausted")
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_exceeded"
        assert body["error"]["message"] == "credits exhausted"
        assert resp.status_code == 429

    def test_sse_error_shape(self):
        sse = _mod._sse_error("boom")
        assert sse.startswith("data: ")
        obj = json.loads(sse[len("data: "):])
        assert obj["error"]["message"] == "boom"
