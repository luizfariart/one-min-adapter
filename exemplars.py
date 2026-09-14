"""Exemplar store + self-improvement logic for the model router.

Pure, dependency-free logic shared by the router (which classifies action vs
text on every request) and the periodic self-improvement job (which learns from
real usage). No network, no FastAPI — safe to import from both.

The router keeps two reference classes — ``action`` (needs a tool: web search,
file/command/email/calendar, real-time data) and ``text`` (pure text work:
format, translate, explain, list). Each class is a list of short exemplar
sentences; a task is classified by cosine similarity of its TF-IDF vector to
each class centroid.

Self-improvement (``improve()``) is driven by a usage log the router appends to
on every request. It:

- **Promotes** recurring tasks: a normalized task text seen at least
  ``promote_threshold`` times in the window becomes an exemplar for the class
  the system actually assigned it to (the ``[[NEED_TOOL]]`` handshake is the
  ground-truth signal that a "text" task was really an "action").
- **Prunes** stale exemplars: any exemplar whose cosine similarity to recent
  usage falls below ``similarity_threshold`` is dropped — it is no longer
  relevant to how the user works.
- **Caps** each class to ``max_per_class`` exemplars, keeping the most frequent
  first, so the index stays small and low-latency.

This runs offline (launchd, daily) — never on the request hot path — so the
per-request cost stays a single cosine comparison.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Seed exemplars. These bootstrap the index before any usage has been seen;
# they are ordinary entries and are pruned like any other once they stop
# matching real usage.
DEFAULT_EXEMPLARS: dict[str, list[str]] = {
    "action": [
        "busque o preço do dólar hoje",
        "pesquise na web sobre isso",
        "leia este arquivo",
        "abra o documento",
        "execute este comando",
        "rode o script",
        "crie um arquivo",
        "salve esta nota",
        "envie um email",
        "agende um lembrete",
        "verifique meu email",
        "cheque o status",
        "baixe este arquivo",
        "instale este pacote",
        "acesse o site",
        "calcule 2 mais 2",
        "mostre a previsão do tempo",
        "qual o valor do bitcoin agora",
        "consulte a cotação atual",
        "o que está acontecendo hoje",
        "me lembre de comprar leite",
        "adicione uma tarefa",
    ],
    "text": [
        "formate em tabela",
        "traduza este texto",
        "resuma isso",
        "explique o que é entropia",
        "por que o céu é azul",
        "me dê ideias de nome",
        "qual a diferença entre vírus e bactéria",
        "liste os itens",
        "corrija a ortografia",
        "escreva um texto sobre",
        "o que significa esta palavra",
        "me dê 3 dicas",
        "reorganize essa lista",
        "corrija a gramática",
    ],
}

# Default locations (overridable via env in the router / job).
EXEMPLARS_PATH = Path.home() / ".hermes" / "one-min-adapter" / "exemplars.json"
USAGE_LOG_PATH = Path.home() / ".hermes" / "one-min-adapter" / "usage.jsonl"

# Self-improvement tuning.
PROMOTE_THRESHOLD = 3        # a task text seen ≥ this many times becomes an exemplar
SIMILARITY_THRESHOLD = 0.4   # cosine below this → exemplar is stale
MAX_PER_CLASS = 60           # keep the index bounded and fast
WINDOW_DAYS = 30             # only recent usage counts
MIN_USAGE = 30               # prune only once there is this much evidence; below
                             # this the log is too thin to call any seed "stale"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zà-ú0-9]+", text.lower())


def _normalize(text: str) -> str:
    """Canonical form for aggregation: lowercase, punctuation→space, collapse
    whitespace, cap length (an exemplar should be short)."""
    text = re.sub(r"[^a-zà-ú0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120]


def _vec(text: str) -> dict[str, float]:
    """Sublinear TF vector (no IDF — stable and deterministic for matching)."""
    tf = Counter(_tokenize(text))
    if not tf:
        return {}
    return {t: 1.0 + math.log(c) for t, c in tf.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def build_centroids(exemplars: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Average TF-IDF vector per class. Returns {class: {token: weight}}."""
    docs = {cls: [_tokenize(d) for d in lst] for cls, lst in exemplars.items()}
    df: Counter[str] = Counter()
    for lst in docs.values():
        for d in lst:
            for tok in set(d):
                df[tok] += 1
    n = sum(len(lst) for lst in docs.values()) or 1
    idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}

    centroids: dict[str, dict[str, float]] = {}
    for cls, lst in docs.items():
        vec: dict[str, float] = {}
        for d in lst:
            for tok, cnt in Counter(d).items():
                vec[tok] = vec.get(tok, 0.0) + (1.0 + math.log(cnt)) * idf.get(tok, 1.0)
        m = len(lst) or 1
        centroids[cls] = {tok: v / m for tok, v in vec.items()}
    return centroids


def load_exemplars(path: Path = EXEMPLARS_PATH) -> dict[str, list[str]]:
    """Load exemplars from disk, falling back to the built-in seeds."""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out: dict[str, list[str]] = {}
            for cls in ("action", "text"):
                vals = data.get(cls)
                if isinstance(vals, list):
                    out[cls] = [str(v) for v in vals if str(v).strip()]
            if out.get("action") or out.get("text"):
                return out
    except Exception:
        pass
    return {k: list(v) for k, v in DEFAULT_EXEMPLARS.items()}


def save_exemplars(exemplars: dict[str, list[str]], path: Path = EXEMPLARS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(exemplars, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def improve(
    exemplars: dict[str, list[str]],
    usage: list[dict[str, Any]],
    *,
    now: float | None = None,
    promote_threshold: int = PROMOTE_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_per_class: int = MAX_PER_CLASS,
    window_days: int = WINDOW_DAYS,
    min_usage: int = MIN_USAGE,
) -> dict[str, list[str]]:
    """Return the improved exemplar set from recent usage. Pure and idempotent.

    ``usage`` entries are ``{"ts": int, "text": str, "action": bool}`` — the
    action flag is the system's final ground truth for that request (index hit
    OR ``[[NEED_TOOL]]`` handshake). Entries with a non-bool ``action`` are
    ignored (they did not pass through the action/text index).
    """
    now = now or time.time()
    cutoff = now - window_days * 86400

    action_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    n_learnable = 0
    for e in usage:
        if not isinstance(e.get("action"), bool):
            continue
        if (e.get("ts") or 0) < cutoff:
            continue
        t = _normalize(e.get("text") or "")
        if not t:
            continue
        n_learnable += 1
        (action_counts if e["action"] else text_counts)[t] += 1

    recent_texts = set(action_counts) | set(text_counts)

    # Cold-start guard: with too little evidence, promote but never prune —
    # a thin log cannot justify dropping a seed exemplar.
    can_prune = n_learnable >= min_usage

    result: dict[str, list[str]] = {}
    for cls, counts in (("action", action_counts), ("text", text_counts)):
        # 1. Keep existing exemplars that still match recent usage.
        kept: list[str] = []
        kept_vecs: dict[str, dict[str, float]] = {}
        for ex in exemplars.get(cls, []):
            exn = _normalize(ex)
            if not exn:
                continue
            v = _vec(exn)
            still_relevant = any(
                _cosine(v, _vec(rt)) >= similarity_threshold for rt in recent_texts
            )
            if not can_prune or still_relevant:
                kept.append(exn)
                kept_vecs[exn] = v

        # 2. Promote recurring task texts not already present.
        seen = set(kept)
        promoted: list[tuple[str, int]] = []
        for t, c in counts.items():
            if c >= promote_threshold and t not in seen:
                promoted.append((t, c))

        # 3. Rank: promoted by frequency; kept survive with a floor score.
        scored: list[tuple[str, float]] = []
        for t, c in promoted:
            scored.append((t, float(c)))
        for t in kept:
            scored.append((t, 1.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        result[cls] = [t for t, _ in scored[:max_per_class]]

    return result
