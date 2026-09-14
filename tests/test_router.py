"""Unit tests for the model router — classification, routing, isolation, thrash.

Pure logic only: no network, no real 1min.ai/Portal calls.
"""

import asyncio
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
def _reset_router_state():
    _mod._reset_thrash_history()
    if hasattr(_mod, "_reset_runtime_caches"):
        _mod._reset_runtime_caches()
    if hasattr(_mod, "_reset_1min_circuit"):
        _mod._reset_1min_circuit()
    yield
    _mod._reset_thrash_history()
    if hasattr(_mod, "_reset_runtime_caches"):
        _mod._reset_runtime_caches()
    if hasattr(_mod, "_reset_1min_circuit"):
        _mod._reset_1min_circuit()


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


class TestRuntimeCaches:
    def test_inconclusive_classification_uses_cache(self, monkeypatch):
        calls = []

        def fake_classifier(text):
            calls.append(text)
            return "fast"

        monkeypatch.setattr(_mod, "_classify_with_model", fake_classifier)
        msg = [{"role": "user", "content": "blorp zigma nebulosa 91827"}]
        assert _mod.classify(msg) == "fast"
        assert _mod.classify(msg) == "fast"
        assert calls == ["blorp zigma nebulosa 91827"]

    def test_needs_tool_uses_cache(self, monkeypatch):
        calls = []

        def fake_centroids():
            calls.append(True)
            return {
                "action": {"leia": 1.0, "arquivo": 1.0},
                "text": {"explique": 1.0},
            }

        monkeypatch.setattr(_mod, "_get_centroids", fake_centroids)
        text = "leia esse arquivo blorpteste 551"
        assert _mod._needs_tool(text) is True
        assert _mod._needs_tool(text) is True
        assert len(calls) == 1

    def test_portal_cache_stores_and_reads_exact_body(self):
        body = {
            "messages": [{"role": "user", "content": "cache portal 441"}],
            "temperature": 0.1,
            "stream": False,
        }
        assert _mod._portal_cacheable(body)
        key = _mod._portal_cache_key(body, "openai/gpt-5.4")
        assert _mod._portal_cache_get(key) is None
        payload = {"id": "abc", "choices": [{"message": {"content": "ok"}}]}
        _mod._portal_cache_put(key, payload)
        assert _mod._portal_cache_get(key) == payload

    def test_portal_cache_skips_stream_and_tool_calls(self):
        assert not _mod._portal_cacheable({
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        })
        assert not _mod._portal_cacheable({
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "web_search"}}],
            "stream": False,
        })


class Test1MinCircuitBreaker:
    def test_open_circuit_skips_model_classification(self, monkeypatch):
        calls = []

        def fake_classifier(text):
            calls.append(text)
            return "fast"

        monkeypatch.setattr(_mod, "_classify_with_model", fake_classifier)
        _mod._record_1min_failure()
        _mod._record_1min_failure()
        msg = [{"role": "user", "content": "mensagem incomum breaker 772"}]
        assert _mod.classify(msg) == "chat"
        assert calls == []

    def test_success_closes_circuit_and_resets_failures(self):
        _mod._record_1min_failure()
        _mod._record_1min_failure()
        assert _mod._is_1min_circuit_open()
        _mod._record_1min_success()
        assert not _mod._is_1min_circuit_open()
        assert _mod._1MIN_CONSECUTIVE_FAILURES == 0
        assert _mod._1MIN_DISABLED_UNTIL == 0.0

    def test_short_prompts_route_to_fast_without_model_call(self, monkeypatch):
        calls = []

        def fake_classifier(text):
            calls.append(text)
            return "chat"

        monkeypatch.setattr(_mod, "_classify_with_model", fake_classifier)
        msg = [{"role": "user", "content": "ok então"}]
        assert _mod.classify(msg) == "fast"
        assert calls == []

    def test_open_circuit_persists_to_disk(self, monkeypatch, tmp_path):
        state_path = tmp_path / "router_state.json"
        monkeypatch.setattr(_mod, "_ROUTER_STATE_PATH", state_path)
        _mod._record_1min_failure()
        _mod._record_1min_failure()
        assert state_path.exists()

        _mod._reset_1min_circuit()
        assert not _mod._is_1min_circuit_open()
        _mod._load_1min_circuit_state()
        assert _mod._is_1min_circuit_open()
        assert _mod._1MIN_CONSECUTIVE_FAILURES >= 2

    def test_success_persists_closed_circuit(self, monkeypatch, tmp_path):
        state_path = tmp_path / "router_state.json"
        monkeypatch.setattr(_mod, "_ROUTER_STATE_PATH", state_path)
        _mod._record_1min_failure()
        _mod._record_1min_failure()
        _mod._record_1min_success()
        _mod._reset_1min_circuit()
        _mod._load_1min_circuit_state()
        assert not _mod._is_1min_circuit_open()
        assert _mod._1MIN_CONSECUTIVE_FAILURES == 0

    def test_probe_does_not_clear_failures_before_threshold(self, monkeypatch, tmp_path):
        state_path = tmp_path / "router_state.json"
        monkeypatch.setattr(_mod, "_ROUTER_STATE_PATH", state_path)
        _mod._record_1min_failure()
        assert not _mod._is_1min_circuit_open()
        assert _mod._1MIN_CONSECUTIVE_FAILURES == 1
        _mod._record_1min_failure()
        assert _mod._is_1min_circuit_open()
        assert _mod._1MIN_CONSECUTIVE_FAILURES == 2


class TestPortalInflightDedup:
    def test_identical_requests_share_one_upstream_call(self, monkeypatch):
        calls = []

        async def fake_fetch(payload):
            calls.append(payload)
            await asyncio.sleep(0.05)
            return {"id": "shared", "choices": [{"message": {"content": "ok"}}]}

        monkeypatch.setattr(_mod, "_fetch_portal_json", fake_fetch)
        body = {
            "messages": [{"role": "user", "content": "dedupe portal 991"}],
            "stream": False,
        }
        payload = _mod._portal_payload(body, "openai/gpt-5.4")

        async def run():
            return await asyncio.gather(
                _mod._get_portal_json(payload, "openai/gpt-5.4"),
                _mod._get_portal_json(payload, "openai/gpt-5.4"),
            )

        results = asyncio.run(run())
        assert len(calls) == 1
        sources = sorted(source for _, source in results)
        assert sources == ["coalesced", "miss"]
        assert results[0][0]["choices"][0]["message"]["content"] == "ok"
        assert results[1][0]["choices"][0]["message"]["content"] == "ok"
