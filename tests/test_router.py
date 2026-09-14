"""Unit tests for the model router — classification, routing, isolation, thrash.

Pure logic only: no network, no real 1min.ai/Portal calls.
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


@pytest.fixture(autouse=True)
def _reset_thrash():
    _mod._reset_thrash_history()
    yield
    _mod._reset_thrash_history()


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

    @pytest.mark.parametrize("text", [
        "explique o que é entropia",
        "por que o céu é azul",
        "me dê 3 ideias de nome",
        "qual a diferença entre vírus e bactéria",
        "como funciona um motor elétrico",
    ])
    def test_mid_keywords(self, text):
        # Simple explanations/questions → mid (cheap), NOT research (expensive).
        assert _mod.classify([{"role": "user", "content": text}]) == "mid"

    def test_empty_falls_back_to_chat(self):
        assert _mod.classify([]) == "chat"

    def test_inconclusive_model_failure_falls_back(self, monkeypatch):
        monkeypatch.setattr(_mod, "_classify_with_model", lambda text: (_ for _ in ()).throw(RuntimeError("down")))
        assert _mod.classify([{"role": "user", "content": "xyz blá"}]) == "chat"


class TestRoutingTable:
    def test_fast_and_mid_go_to_1min(self):
        assert _mod.ROUTE_TABLE["fast"]["backend"] == "1min"
        assert _mod.ROUTE_TABLE["mid"]["backend"] == "1min"

    def test_heavy_tasks_go_to_portal(self):
        for cls in ("code", "research", "chat", "review"):
            assert _mod.ROUTE_TABLE[cls]["backend"] == "portal"

    def test_light_tasks_have_cheap_fallback(self):
        for cls in ("fast", "mid"):
            fb = _mod.ROUTE_TABLE[cls].get("fallback_model")
            assert fb and "flash" in fb


class TestIsolatedPrompt:
    def test_task_only_no_system(self):
        msgs = [
            {"role": "system", "content": "INSTRUÇÕES GIGANTES AQUI"},
            {"role": "user", "content": "formate em tabela: A B. 1 2"},
        ]
        out = _mod._isolated_prompt(msgs)
        assert "INSTRUÇÕES GIGANTES" not in out  # system prompt stripped
        assert "formate em tabela" in out          # the task is kept

    def test_followup_keeps_previous_answer(self):
        msgs = [
            {"role": "user", "content": "traduza bom dia para inglês"},
            {"role": "assistant", "content": "good morning"},
            {"role": "user", "content": "e agora boa noite"},
        ]
        out = _mod._isolated_prompt(msgs)
        assert "good morning" in out
        assert "boa noite" in out

    def test_minimal_drops_context(self):
        msgs = [
            {"role": "user", "content": "traduza bom dia"},
            {"role": "assistant", "content": "good morning"},
            {"role": "user", "content": "e boa noite"},
        ]
        out = _mod._isolated_prompt(msgs, minimal=True)
        assert "good morning" not in out  # even follow-up context dropped
        assert "boa noite" in out

    def test_empty_returns_empty(self):
        assert _mod._isolated_prompt([]) == ""


class TestThrashDetection:
    def test_no_thrash_when_stable(self):
        for _ in range(6):
            _mod._record_backend("portal")
        assert not _mod._detect_thrash()

    def test_thrash_when_bouncing(self):
        for b in ("portal", "1min", "portal", "1min", "portal", "1min"):
            _mod._record_backend(b)
        assert _mod._detect_thrash()

    def test_no_thrash_with_few_requests(self):
        _mod._record_backend("portal")
        _mod._record_backend("1min")
        assert not _mod._detect_thrash()  # < 4 requests → not enough history


class TestToolNeedDetection:
    @pytest.mark.parametrize("text", [
        "busque o preço do dólar hoje",
        "leia esse arquivo",
        "execute o comando de build",
        "envie um email para o cliente",
        "agende um lembrete amanhã",
        "qual a cotação do bitcoin agora",
        "mostre a previsão do tempo",
        "salve essa nota no arquivo",
        "me lembre de comprar leite",
    ])
    def test_action_tasks_detected(self, text):
        assert _mod._needs_tool(text)

    @pytest.mark.parametrize("text", [
        "formate em tabela",
        "traduza isso para inglês",
        "explique o que é entropia",
        "me dê 3 ideias de nome",
        "resuma esse texto",
        "corrija a ortografia",
    ])
    def test_text_tasks_not_action(self, text):
        assert not _mod._needs_tool(text)

    def test_empty_text_not_action(self):
        assert not _mod._needs_tool("")


class TestPortalPayload:
    def test_preserves_tools_and_tool_choice(self):
        body = {
            "model": "ignored",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
            "tool_choice": "auto",
            "temperature": 0.7,
            "stream": True,
        }
        out = _mod._portal_payload(body, "deepseek/deepseek-v4-pro")
        assert out["model"] == "deepseek/deepseek-v4-pro"  # overridden
        assert out["tools"] == body["tools"]                # preserved
        assert out["tool_choice"] == "auto"                 # preserved
        assert out["temperature"] == 0.7                    # preserved
        assert out["stream"] is True                        # preserved

    def test_does_not_mutate_original(self):
        body = {"model": "x", "messages": [], "tools": []}
        _mod._portal_payload(body, "y")
        assert body["model"] == "x"  # original untouched


class TestIsolatedPromptHasNeedToolHandshake:
    def test_need_tool_instruction_present(self):
        out = _mod._isolated_prompt([{"role": "user", "content": "formate em tabela"}])
        assert _mod.NEED_TOOL in out

    def test_need_tool_marker_constant(self):
        assert _mod.NEED_TOOL == "[[NEED_TOOL]]"
