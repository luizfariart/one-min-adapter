"""Unit tests for the exemplar store + self-improvement logic.

Pure logic only — no network, no filesystem side effects beyond temp dirs.
"""

import importlib.util
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "exemplars",
    Path(__file__).resolve().parent.parent / "exemplars.py",
)
_ex = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ex)


def _usage(entries):
    """Build usage entries from (text, action) pairs at a recent timestamp."""
    now = int(time.time())
    return [
        {"ts": now, "text": t, "action": a}
        for t, a in entries
    ]


class TestNormalize:
    def test_lowercase_strip_punct(self):
        assert _ex._normalize("Busque  o  Preço, HOJE!") == "busque o preço hoje"

    def test_truncates_long_text(self):
        long = "palavra " * 100
        assert len(_ex._normalize(long)) <= 120


class TestImprove:
    def test_promotes_recurring_action(self):
        usage = _usage([
            ("agende um lembrete para amanhã", True),
            ("agende um lembrete para amanhã", True),
            ("agende um lembrete para amanhã", True),
        ])
        before = {"action": [], "text": []}
        after = _ex.improve(before, usage, promote_threshold=3)
        assert "agende um lembrete para amanhã" in after["action"]

    def test_promotes_recurring_text(self):
        usage = _usage([
            ("formate essa lista", False),
            ("formate essa lista", False),
            ("formate essa lista", False),
        ])
        after = _ex.improve({"action": [], "text": []}, usage, promote_threshold=3)
        assert "formate essa lista" in after["text"]

    def test_below_threshold_not_promoted(self):
        usage = _usage([
            ("traduza isso", False),
            ("traduza isso", False),
        ])
        after = _ex.improve({"action": [], "text": []}, usage, promote_threshold=3)
        assert after["text"] == []

    def test_prunes_stale_exemplar(self):
        # Exemplar "abra o documento" has no matching recent usage → pruned.
        before = {"action": ["abra o documento"], "text": []}
        usage = _usage([
            ("formate em tabela", False),
            ("formate em tabela", False),
            ("formate em tabela", False),
        ])
        after = _ex.improve(before, usage, promote_threshold=3, min_usage=1)
        assert "abra o documento" not in after["action"]

    def test_keeps_exemplar_matching_recent_usage(self):
        before = {"action": ["leia este arquivo"], "text": []}
        usage = _usage([
            ("leia este arquivo por favor", True),
            ("leia esse arquivo agora", True),
        ])
        after = _ex.improve(before, usage, promote_threshold=99, min_usage=1)
        assert "leia este arquivo" in after["action"]

    def test_cold_start_does_not_prune(self):
        # Few entries (< min_usage) → seeds survive even if nothing matches.
        before = {"action": ["abra o documento"], "text": ["formate em tabela"]}
        usage = _usage([
            ("traduza isso", False),
            ("traduza isso", False),
            ("traduza isso", False),
        ])
        after = _ex.improve(before, usage, promote_threshold=3, min_usage=30)
        assert "abra o documento" in after["action"]
        assert "formate em tabela" in after["text"]

    def test_ignores_non_bool_action(self):
        # Heavy tasks (action=None) must not pollute the index.
        usage = _usage([
            ("escreva código python", None),
            ("escreva código python", None),
            ("escreva código python", None),
        ])
        after = _ex.improve({"action": [], "text": []}, usage, promote_threshold=3)
        assert after["action"] == [] and after["text"] == []

    def test_caps_per_class(self):
        usage = _usage([(f"tarefa de texto {i}", False) for i in range(10) for _ in range(3)])
        after = _ex.improve({"action": [], "text": []}, usage, promote_threshold=3, max_per_class=5)
        assert len(after["text"]) == 5

    def test_ignores_old_usage(self):
        old = int(time.time()) - 40 * 86400
        usage = [
            {"ts": old, "text": "traduza isso", "action": False},
            {"ts": old, "text": "traduza isso", "action": False},
            {"ts": old, "text": "traduza isso", "action": False},
        ]
        after = _ex.improve({"action": [], "text": []}, usage, promote_threshold=3)
        assert after["text"] == []


class TestCentroids:
    def test_builds_action_and_text(self):
        c = _ex.build_centroids(_ex.DEFAULT_EXEMPLARS)
        assert "action" in c and "text" in c
        assert c["action"] and c["text"]  # non-empty vectors

    def test_cosine_identical_is_one(self):
        v = _ex._vec("formate em tabela")
        assert abs(_ex._cosine(v, v) - 1.0) < 1e-9

    def test_cosine_disjoint_is_zero(self):
        assert _ex._cosine(_ex._vec("aaa bbb"), _ex._vec("ccc ddd")) == 0.0


class TestLoadSave:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "exemplars.json"
        data = {"action": ["leia arquivo"], "text": ["formate tabela"]}
        _ex.save_exemplars(data, p)
        assert _ex.load_exemplars(p) == data

    def test_load_missing_falls_back_to_defaults(self, tmp_path):
        p = tmp_path / "nope.json"
        got = _ex.load_exemplars(p)
        assert got["action"] == _ex.DEFAULT_EXEMPLARS["action"]

    def test_load_corrupt_falls_back(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{ not json", encoding="utf-8")
        got = _ex.load_exemplars(p)
        assert got["action"] == _ex.DEFAULT_EXEMPLARS["action"]
