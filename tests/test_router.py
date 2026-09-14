"""Unit tests for the model router — classification, routing, and fallback logic.

Pure logic only: no network, no real 1min.ai/Portal calls. The classification
rules and route table are the behaviour contract.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "model_router",
    Path(__file__).resolve().parent.parent / "model_router.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)


class TestClassification:
    @pytest.mark.parametrize("text", [
        "escreva uma função python",
        "debuga esse bug",
        "refatora esse código",
        "tem um erro de sql aqui",
    ])
    def test_code_keywords(self, text):
        assert _mod.classify([{"role": "user", "content": text}]) == "code"

    @pytest.mark.parametrize("text", [
        "pesquisa sobre energia solar",
        "compara esses dois artigos",
        "analisa os dados",
        "por que o céu é azul",
    ])
    def test_research_keywords(self, text):
        assert _mod.classify([{"role": "user", "content": text}]) == "research"

    @pytest.mark.parametrize("text", [
        "formate em tabela",
        "resume esse texto",
        "traduz isso para inglês",
        "lista os itens",
    ])
    def test_fast_keywords(self, text):
        assert _mod.classify([{"role": "user", "content": text}]) == "fast"

    def test_empty_falls_back_to_chat(self):
        assert _mod.classify([]) == "chat"
        assert _mod.classify([{"role": "user", "content": ""}]) == "chat"

    def test_inconclusive_returns_something_valid(self, monkeypatch):
        # No keyword matches → falls through to the model classifier; if that
        # fails too, it must still return a valid class (never raise).
        monkeypatch.setattr(_mod, "_classify_with_model", lambda text: "chat")
        result = _mod.classify([{"role": "user", "content": "blá blá blá xyz"}])
        assert result in _mod.ROUTE_TABLE

    def test_inconclusive_model_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(_mod, "_classify_with_model", lambda text: (_ for _ in ()).throw(RuntimeError("down")))
        result = _mod.classify([{"role": "user", "content": "xyz"}])
        assert result == "chat"


class TestRoutingTable:
    def test_fast_goes_to_1min(self):
        assert _mod.ROUTE_TABLE["fast"]["backend"] == "1min"

    def test_heavy_tasks_go_to_portal(self):
        for cls in ("code", "research", "chat", "review"):
            assert _mod.ROUTE_TABLE[cls]["backend"] == "portal"

    def test_fast_has_cheap_fallback(self):
        fb = _mod.ROUTE_TABLE["fast"].get("fallback_model")
        assert fb and "flash" in fb

    def test_every_class_has_a_model(self):
        for cls, route in _mod.ROUTE_TABLE.items():
            assert route.get("model"), f"{cls} missing model"


class TestLastUserText:
    def test_gets_last_user_message(self):
        msgs = [
            {"role": "user", "content": "primeiro"},
            {"role": "assistant", "content": "resposta"},
            {"role": "user", "content": "segundo"},
        ]
        assert _mod._last_user_text(msgs) == "segundo"

    def test_empty_when_no_user_message(self):
        assert _mod._last_user_text([{"role": "assistant", "content": "oi"}]) == ""

    def test_flattens_multimodal_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "veja isto"}]}]
        assert _mod._last_user_text(msgs) == "veja isto"
