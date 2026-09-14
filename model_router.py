"""Model router — dynamic per-request routing with free classification.

A local OpenAI-compatible proxy that sits in front of the Hermes agent. On
every `/v1/chat/completions` request it:

1. **Classifies** the task from the conversation (deterministic keyword rules
   first, then a free/cheap model — 1min.ai deepseek-flash — only when the
   rules are inconclusive).
2. **Detects tool need** with a vector index (TF-IDF + cosine similarity):
   tasks that require an action (web search, file/command/email/calendar,
   real-time data) are routed to the Portal, which supports function calling.
3. **Routes** the request to the right backend:
   - ``fast`` / ``mid`` → 1min.ai directly (pre-paid credit balance), with an
     **isolated prompt** — only the task is sent, not the huge system prompt or
     history, so the credit spend is minimal ("a fresh chat" per task).
   - ``code`` / ``research`` / ``chat`` / ``review`` / any *action* task → the
     Hermes portal proxy (``http://127.0.0.1:8645/v1``), which attaches the
     Nous OAuth credentials. The **full request body is forwarded** (tools,
     tool_choice, temperature, ...) so the Portal can actually call tools.
4. **Falls back** cleanly: if the primary backend fails (credits exhausted,
   rate limit, timeout), it routes to the other backend. It never hangs and
   never returns a half-formed 200.

**`[[NEED_TOOL]]` handshake:** the 1min.ai prompt tells the cheap model to
reply with exactly ``[[NEED_TOOL]]`` when the task requires an external action
it cannot perform from text alone. The router detects that marker and re-routes
the *original* request (with its tools) to the Portal — so a lightweight task
that unexpectedly needs a tool is never left hanging.

**Thrash detection:** the router keeps a rolling history of the backend used
per request. When the conversation is bouncing between Portal and 1min.ai
frequently (which would otherwise re-send the big Portal prompt on every heavy
turn), the router switches lightweight tasks to *minimal* isolation — dropping
even the follow-up context — to keep spend at the absolute minimum while the
heavy turns keep their cached Portal prefix.
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from pathlib import Path
import asyncio

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from exemplars import (  # noqa: E402
    USAGE_LOG_PATH,
    build_centroids,
    load_exemplars,
    _normalize,
    _vec,
    _cosine,
)

log = logging.getLogger("model_router")

# Backends
ONEMIN_API_BASE = "https://api.1min.ai"
ONEMIN_CHAT_PATH = "/api/chat-with-ai"
PORTAL_PROXY_URL = os.environ.get("PORTAL_PROXY_URL", "http://127.0.0.1:8645/v1")

ONEMIN_REQUEST_TIMEOUT = 5   # keep fallback snappy when 1min stalls
PORTAL_REQUEST_TIMEOUT = 30  # allow normal model latency without false 502s
CLASSIFY_TIMEOUT = 3         # the classifier should never add double-digit latency

CLASSIFY_CACHE_TTL_S = 600
TOOL_NEED_CACHE_TTL_S = 600
PORTAL_CACHE_TTL_S = 45
RUNTIME_CACHE_MAX_ENTRIES = 256

ONEMIN_FAILURE_THRESHOLD = 2
ONEMIN_CIRCUIT_OPEN_S = 180
_1MIN_CONSECUTIVE_FAILURES = 0
_1MIN_DISABLED_UNTIL = 0.0

_classify_cache: dict[str, tuple[float, str]] = {}
_tool_need_cache: dict[str, tuple[float, bool]] = {}
_portal_response_cache: dict[str, tuple[float, dict]] = {}
_portal_inflight: dict[str, asyncio.Future] = {}

_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
_ROUTER_STATE_PATH = Path(os.environ.get("ROUTER_STATE_PATH", str(_HERMES_HOME / "one-min-adapter" / "router_state.json")))

# Marker the 1min model must emit when the task needs an external tool/action.
NEED_TOOL = "[[NEED_TOOL]]"

# Portal model used for tool-requiring tasks and NEED_TOOL re-routes.
PORTAL_TOOL_MODEL = "deepseek/deepseek-v4-pro"

# Task classes and their target backend + model.
# "fast" and "mid" go to 1min.ai (credit balance); the rest go to the Portal.
ROUTE_TABLE: dict[str, dict] = {
    "fast":     {"backend": "1min",   "model": "deepseek-flash", "fallback_model": "deepseek/deepseek-v4-flash"},
    "mid":      {"backend": "1min",   "model": "deepseek-chat",  "fallback_model": "deepseek/deepseek-v4-flash"},
    "code":     {"backend": "portal", "model": "deepseek/deepseek-v4-pro"},
    "research": {"backend": "portal", "model": "anthropic/claude-opus-5"},
    "chat":     {"backend": "portal", "model": "openai/gpt-5.4"},
    "review":   {"backend": "portal", "model": "kwaipilot/kat-coder-pro-v2.5"},
}

# Keyword rules → task class. Order matters (first match wins).
# Deterministic and free — most requests never touch the classifier model.
# "mid" comes before "research" so simple explanations (which used to go to the
# expensive claude-opus) now land on the cheap 1min backend instead.
_KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("code", (
        "code", "codigo", "código", "debug", "debugar", "script", "função", "funcao",
        "bug", "refactor", "refatorar", "teste", "test", "api", "classe", "class",
        "implementar", "implement", "programa", "programar", "python", "typescript",
        "javascript", "sql", "regex", "compilar", "build", "commit", "git",
    )),
    ("review", (
        "review", "revisar", "revisão", "revisao", "auditar", "audit", "validar",
        "validate", "code review", "inspecionar", "inspect",
    )),
    ("mid", (
        "explique", "explic", "o que é", "o que sao", "o que são", "o que e",
        "por que", "porque", "como funciona", "como faço", "como faco", "como fazer",
        "qual a diferença", "qual a diferenca", "diferença entre", "diferenca entre",
        "ideias", "ideia", "dicas", "exemplos", "exemplo", "sugest",
        "me dê", "me de", "me diga", "me diga", "escreva um texto", "elabore",
        "o que significa", "o que quer dizer",
    )),
    ("research", (
        "pesquis", "research", "analis", "compare", "compar", "sintetiz",
        "fonte", "referência", "referencia", "estud", "artigo", "paper", "academic",
        "busca", "search", "levantamento", "deep research",
    )),
    ("fast", (
        "formata", "formatar", "formate", "tabela", "lista",
        "traduz", "traduza", "resum", "classif", "uma palavra",
        "corrigir ortografia", "upper", "lower", "trim", "ordenar", "orden",
    )),
]


# --------------------------------------------------------------------------- #
# Vector index — TF-IDF + cosine similarity to detect "action" vs "text" tasks
# --------------------------------------------------------------------------- #
#
# The exemplars and centroids live in exemplars.py (shared with the daily
# self-improvement job). The router hot-reloads them when exemplars.json changes,
# so newly learned exemplars take effect without a restart.

_EXEMPLARS_PATH = Path.home() / ".hermes" / "one-min-adapter" / "exemplars.json"
_exemplars_state: dict = {"mtime": None, "centroids": None}


def _prune_expired(cache: dict) -> None:
    now = time.time()
    expired = [k for k, (until, _) in cache.items() if until <= now]
    for k in expired:
        cache.pop(k, None)


def _cache_get(cache: dict, key: str):
    item = cache.get(key)
    if not item:
        return None
    until, value = item
    if until <= time.time():
        cache.pop(key, None)
        return None
    return value


def _cache_put(cache: dict, key: str, value, ttl_s: int) -> None:
    _prune_expired(cache)
    if len(cache) >= RUNTIME_CACHE_MAX_ENTRIES:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
    cache[key] = (time.time() + ttl_s, value)


def _json_clone(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _router_state_payload() -> dict:
    return {
        "one_min_consecutive_failures": _1MIN_CONSECUTIVE_FAILURES,
        "one_min_disabled_until": _1MIN_DISABLED_UNTIL,
    }


def _save_1min_circuit_state() -> None:
    try:
        _ROUTER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _ROUTER_STATE_PATH.with_suffix(_ROUTER_STATE_PATH.suffix + ".tmp")
        tmp.write_text(json.dumps(_router_state_payload(), ensure_ascii=False), encoding="utf-8")
        tmp.replace(_ROUTER_STATE_PATH)
    except Exception:
        pass


def _load_1min_circuit_state() -> None:
    global _1MIN_CONSECUTIVE_FAILURES, _1MIN_DISABLED_UNTIL
    try:
        if not _ROUTER_STATE_PATH.exists():
            return
        data = json.loads(_ROUTER_STATE_PATH.read_text(encoding="utf-8"))
        _1MIN_CONSECUTIVE_FAILURES = max(0, int(data.get("one_min_consecutive_failures", 0)))
        _1MIN_DISABLED_UNTIL = max(0.0, float(data.get("one_min_disabled_until", 0.0)))
        if _1MIN_DISABLED_UNTIL <= time.time():
            _1MIN_CONSECUTIVE_FAILURES = 0
            _1MIN_DISABLED_UNTIL = 0.0
    except Exception:
        _1MIN_CONSECUTIVE_FAILURES = 0
        _1MIN_DISABLED_UNTIL = 0.0


def _reset_runtime_caches() -> None:
    _classify_cache.clear()
    _tool_need_cache.clear()
    _portal_response_cache.clear()
    _portal_inflight.clear()


def _record_1min_failure() -> None:
    global _1MIN_CONSECUTIVE_FAILURES, _1MIN_DISABLED_UNTIL
    _1MIN_CONSECUTIVE_FAILURES += 1
    if _1MIN_CONSECUTIVE_FAILURES >= ONEMIN_FAILURE_THRESHOLD:
        _1MIN_DISABLED_UNTIL = time.time() + ONEMIN_CIRCUIT_OPEN_S
    _save_1min_circuit_state()


def _record_1min_success() -> None:
    global _1MIN_CONSECUTIVE_FAILURES, _1MIN_DISABLED_UNTIL
    _1MIN_CONSECUTIVE_FAILURES = 0
    _1MIN_DISABLED_UNTIL = 0.0
    _save_1min_circuit_state()


def _is_1min_circuit_open() -> bool:
    if _1MIN_DISABLED_UNTIL > time.time():
        return True
    if _1MIN_DISABLED_UNTIL != 0.0:
        _record_1min_success()
    return False


def _reset_1min_circuit() -> None:
    global _1MIN_CONSECUTIVE_FAILURES, _1MIN_DISABLED_UNTIL
    _1MIN_CONSECUTIVE_FAILURES = 0
    _1MIN_DISABLED_UNTIL = 0.0


def _portal_cacheable(body: dict) -> bool:
    return not body.get("stream") and not body.get("tools")


def _portal_cache_key(body: dict, model: str) -> str:
    digest = hashlib.sha256(
        json.dumps({"model": model, "body": body}, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def _portal_cache_get(key: str) -> dict | None:
    value = _cache_get(_portal_response_cache, key)
    return _json_clone(value) if value is not None else None


def _portal_cache_put(key: str, value: dict) -> None:
    _cache_put(_portal_response_cache, key, _json_clone(value), PORTAL_CACHE_TTL_S)


_SHORT_FAST_LEAD_WORDS = {
    "ok", "beleza", "blz", "segue", "continua", "continue", "sim", "não", "nao",
    "valeu", "obrigado", "manda", "pode", "certo", "show",
}


def _is_short_low_stakes_prompt(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 3 and len(text) <= 24 and words[0] in _SHORT_FAST_LEAD_WORDS


def _get_centroids() -> dict[str, dict[str, float]]:
    """Return centroids, rebuilding them only when exemplars.json changed.

    One os.path.getmtime() per call is ~microseconds — negligible on the hot
    path, but it lets the daily job's write take effect immediately.
    """
    try:
        mtime = _EXEMPLARS_PATH.stat().st_mtime if _EXEMPLARS_PATH.exists() else None
    except OSError:
        mtime = None
    if mtime != _exemplars_state["mtime"] or _exemplars_state["centroids"] is None:
        _exemplars_state["centroids"] = build_centroids(load_exemplars(_EXEMPLARS_PATH))
        _exemplars_state["mtime"] = mtime
        _tool_need_cache.clear()
    return _exemplars_state["centroids"]


def _needs_tool(text: str) -> bool:
    """True when the task vector is closer to the 'action' centroid than 'text'."""
    norm = _normalize(text)
    cached = _cache_get(_tool_need_cache, norm)
    if cached is not None:
        return cached

    vec = _vec(text)
    if not vec:
        return False
    centroids = _get_centroids()
    result = _cosine(vec, centroids["action"]) > _cosine(vec, centroids["text"])
    _cache_put(_tool_need_cache, norm, result, TOOL_NEED_CACHE_TTL_S)
    return result


def _log_usage(text: str, action: bool | None) -> None:
    """Append one usage record for the daily self-improvement job.

    ``action`` is the system's ground truth: True (routed to Portal for a tool),
    False (answered by 1min as pure text), or None (heavy task that never went
    through the action/text index — not learnable). Best-effort: logging must
    never break a request, and never block it for more than a few µs.
    """
    try:
        record = json.dumps(
            {"ts": int(time.time()), "text": _normalize(text), "action": action},
            ensure_ascii=False,
        )
        with open(USAGE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(record + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Thrash detection — rolling history of backend per request
# --------------------------------------------------------------------------- #

_RECENT_BACKENDS: list[str] = []
_MAX_HISTORY = 8
# A "switch" is a backend change between two consecutive requests. When switches
# dominate the recent window, the conversation is thrashing between Portal and
# 1min.ai and we isolate lightweight tasks harder to protect the Portal cache.
_THRASH_SWITCH_RATIO = 0.5


def _record_backend(backend: str) -> None:
    _RECENT_BACKENDS.append(backend)
    if len(_RECENT_BACKENDS) > _MAX_HISTORY:
        _RECENT_BACKENDS.pop(0)


def _detect_thrash() -> bool:
    """True when the recent request history is bouncing between backends."""
    if len(_RECENT_BACKENDS) < 4:
        return False
    switches = sum(1 for a, b in zip(_RECENT_BACKENDS, _RECENT_BACKENDS[1:]) if a != b)
    return switches >= int(len(_RECENT_BACKENDS) * _THRASH_SWITCH_RATIO)


def _reset_thrash_history() -> None:
    _RECENT_BACKENDS.clear()


# --------------------------------------------------------------------------- #
# Key + message helpers
# --------------------------------------------------------------------------- #

def _api_key() -> str:
    env = os.environ.get("ONEMIN_API_KEY", "")
    if env:
        return env.strip()
    key_file = os.environ.get("ONEMIN_KEY_FILE", str(Path.home() / ".hermes/secrets/1min.key"))
    try:
        p = Path(key_file)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content") or ""
            if isinstance(c, str):
                return c
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def _looks_portuguese(text: str) -> bool:
    """Cheap heuristic so isolated prompts keep the user's language."""
    pt_markers = ("ç", "ã", "õ", "é", "á", "í", "ó", "ú", "ê", "ô", "à", " para ", " como ", " que ", " um ", " uma ")
    return any(m in text for m in pt_markers)


def classify(messages: list[dict]) -> str:
    """Classify the task. Deterministic rules first (free), then the cheap
    1min.ai model only if inconclusive. Never raises; falls back to 'chat'."""
    text = _last_user_text(messages).lower().strip()
    if not text:
        return "chat"

    for cls, kws in _KEYWORD_RULES:
        if any(kw in text for kw in kws):
            return cls

    if _is_short_low_stakes_prompt(text):
        return "fast"

    cached = _cache_get(_classify_cache, text)
    if cached is not None:
        return cached

    if _is_1min_circuit_open():
        return "chat"

    try:
        result = _classify_with_model(text)
    except Exception:
        result = "chat"
    _cache_put(_classify_cache, text, result, CLASSIFY_CACHE_TTL_S)
    return result


def _classify_with_model(text: str) -> str:
    """Free/cheap classification via 1min.ai deepseek-flash."""
    if _is_1min_circuit_open():
        return "chat"
    key = _api_key()
    if not key:
        return "chat"
    prompt = (
        "Classify this task into exactly one category: code, research, chat, "
        "review, mid, or fast. Reply with only that single word.\n\n"
        f"Task: {text[:500]}"
    )
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": "deepseek-flash",
        "promptObject": {"prompt": prompt},
    }
    try:
        with httpx.Client(timeout=CLASSIFY_TIMEOUT) as client:
            resp = client.post(
                f"{ONEMIN_API_BASE}{ONEMIN_CHAT_PATH}",
                headers={"API-KEY": key, "Content-Type": "application/json"},
                json=payload,
            )
        resp.raise_for_status()
        record = resp.json().get("aiRecord") or {}
        detail = record.get("aiRecordDetail") or {}
        result = detail.get("resultObject") or []
        answer = "".join(str(x) for x in result).strip().lower()
    except Exception:
        _record_1min_failure()
        return "chat"

    _record_1min_success()
    for cls in ROUTE_TABLE:
        if cls in answer:
            return cls
    return "chat"


# --------------------------------------------------------------------------- #
# Prompt builders — full (Portal) vs isolated (1min lightweight)
# --------------------------------------------------------------------------- #

def _build_prompt(messages: list[dict]) -> str:
    """Full prompt (system + history) — used for Portal forwarding context."""
    system = []
    parts = []
    for m in messages or []:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system.append(content)
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    out = []
    if system:
        out.append("System instructions:\n" + "\n".join(system))
    out.extend(parts)
    return "\n\n".join(out)


_NEED_TOOL_INSTRUCTION = (
    "\n\nIf this task requires an external action or tool you cannot perform "
    "from the text alone (searching the web, reading/writing files, running a "
    "command, sending email, accessing a calendar or real-time data), reply "
    f"with exactly {NEED_TOOL} and nothing else."
)


def _isolated_prompt(messages: list[dict], *, minimal: bool = False) -> str:
    """Prompt for lightweight (fast/mid) tasks — the task alone, without the
    huge system prompt or full history. This is the "fresh chat" optimization:
    a formatting/explanation task does not need the agent's system prompt.

    Carries a ``[[NEED_TOOL]]`` handshake instruction so that, if the cheap
    model realizes the task actually requires a tool, it signals the router to
    re-route to the Portal instead of answering with a dead-end string.

    When ``minimal`` is False, a short follow-up (≤15 words) gets the previous
    assistant reply as context so "and in English?" still refers to something.
    When ``minimal`` is True (thrash mode), even that context is dropped to
    keep every credit.
    """
    last = _last_user_text(messages)
    if not last:
        return ""

    # Language guard: isolated prompts skip the system prompt that normally
    # forces Portuguese, so re-assert it here when the task is in Portuguese.
    lang = "Respond in the same language as the task.\n\n" if _looks_portuguese(last) else ""

    body = ""
    if not minimal and len(last.split()) <= 15:
        for m in reversed(messages or []):
            if m.get("role") == "assistant":
                prev = (m.get("content") or "").strip()
                if prev:
                    body = f"Context (previous answer):\n{prev[:1000]}\n\nNew task:\n"
                break
    return f"{lang}{body}{last}{_NEED_TOOL_INSTRUCTION}"


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

_ONEMIN_MODEL_MAP = {
    "deepseek-flash": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-chat": "deepseek-chat",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
}


def _openai_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": code, "message": message}})


def _call_1min(prompt: str, model: str) -> dict:
    """Call 1min.ai with an already-built prompt string.

    Synchronous but called via ``asyncio.to_thread`` from the async handler so
    it never blocks the uvicorn event loop — critical for concurrent routing."""
    if _is_1min_circuit_open():
        raise RuntimeError("1min circuit open")
    key = _api_key()
    if not key:
        raise ValueError("1min.ai API key not configured")
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": _ONEMIN_MODEL_MAP.get(model, model),
        "promptObject": {"prompt": prompt, "settings": {"webSearchSettings": {"webSearch": False}}},
    }
    try:
        with httpx.Client(timeout=ONEMIN_REQUEST_TIMEOUT) as client:
            resp = client.post(
                f"{ONEMIN_API_BASE}{ONEMIN_CHAT_PATH}",
                headers={"API-KEY": key, "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code == 401:
            raise PermissionError("1min.ai returned 401")
        if resp.status_code == 429:
            raise RuntimeError("1min.ai returned 429 — rate limited or credits exhausted")
        resp.raise_for_status()
        record = resp.json().get("aiRecord") or {}
        if record.get("status") == "FAILED":
            raise RuntimeError("1min.ai request FAILED")
    except Exception:
        _record_1min_failure()
        raise

    _record_1min_success()
    return record


def _extract_text(record: dict) -> str:
    detail = record.get("aiRecordDetail") or {}
    result = detail.get("resultObject") or []
    if isinstance(result, list):
        return "".join(str(x) for x in result)
    return str(result)


def _portal_payload(body: dict, model: str) -> dict:
    """Build the upstream Portal payload from the original Hermes request,
    preserving tools/tool_choice/temperature/etc. and overriding only model."""
    payload = dict(body)
    payload["model"] = model
    return payload


async def _fetch_portal_json(payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=PORTAL_REQUEST_TIMEOUT) as client:
        upstream = await client.post(
            f"{PORTAL_PROXY_URL}/chat/completions",
            headers={"Authorization": "Bearer hermes-router"},
            json=payload,
        )

    try:
        data = upstream.json()
    except Exception:
        raise RuntimeError(f"portal proxy HTTP {upstream.status_code}, non-JSON body") from None
    if upstream.status_code >= 400:
        raise RuntimeError(json.dumps(data, ensure_ascii=False))
    return data


async def _get_portal_json(payload: dict, model: str) -> tuple[dict, str]:
    cacheable = _portal_cacheable(payload)
    if not cacheable:
        return await _fetch_portal_json(payload), "bypass"

    cache_key = _portal_cache_key(payload, model)
    cached = _portal_cache_get(cache_key)
    if cached is not None:
        return cached, "hit"

    inflight = _portal_inflight.get(cache_key)
    if inflight is not None:
        data = await asyncio.shield(inflight)
        return _json_clone(data), "coalesced"

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _portal_inflight[cache_key] = future
    try:
        data = await _fetch_portal_json(payload)
        _portal_cache_put(cache_key, data)
        future.set_result(_json_clone(data))
        return data, "miss"
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        _portal_inflight.pop(cache_key, None)


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

_load_1min_circuit_state()

app = FastAPI(title="Hermes model router")


@app.get("/health")
def health():
    return {
        "ok": True,
        "1min_key_configured": bool(_api_key()),
        "portal_proxy": PORTAL_PROXY_URL,
        "thrash": _detect_thrash(),
        "1min_circuit_open": _is_1min_circuit_open(),
        "1min_disabled_for_s": round(max(0.0, _1MIN_DISABLED_UNTIL - time.time()), 3),
        "cache_entries": {
            "classify": len(_classify_cache),
            "tool_need": len(_tool_need_cache),
            "portal": len(_portal_response_cache),
        },
        "portal_inflight": len(_portal_inflight),
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "router"}
            for model in ROUTE_TABLE
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    stream = bool(body.get("stream", False))
    user_text = _last_user_text(messages)
    # classify() may call the network; run it in a thread so the event loop stays unblocked
    task = await asyncio.to_thread(classify, messages)
    thrash = _detect_thrash()
    log.info("routing -> %s (thrash=%s, stream=%s)", task, thrash, stream)

    # Action tasks (need a tool) → Portal, regardless of cheap task class. The
    # vector index catches cases keywords miss ("cotação do dólar agora",
    # "leia esse arquivo") and routes them to the tool-capable backend.
    if task in ("fast", "mid") and _needs_tool(user_text):
        log.info("action detected → portal (tool required)")
        _record_backend("portal")
        _log_usage(user_text, action=True)
        return await _forward_portal(body, PORTAL_TOOL_MODEL)

    route = ROUTE_TABLE.get(task, ROUTE_TABLE["chat"])

    # Lightweight (fast/mid) → 1min.ai with an ISOLATED prompt (task only, no
    # system prompt/history) so the credit spend stays minimal. In thrash mode
    # even the follow-up context is dropped. ANY failure falls back to Portal.
    if route["backend"] == "1min":
        prompt = _isolated_prompt(messages, minimal=thrash)
        try:
            # _call_1min is synchronous (httpx.Client) — run in a thread so it
            # never blocks the async event loop, even under network timeouts.
            record = await asyncio.to_thread(_call_1min, prompt, route["model"])
        except Exception as exc:
            log.warning("1min backend failed (%s) — falling back to portal", exc)
            _record_backend("portal")
            _log_usage(user_text, action=None)  # fallback — not a clean signal
            return await _forward_portal(body, route.get("fallback_model") or "deepseek/deepseek-v4-flash")
        text = _extract_text(record)

        # NEED_TOOL handshake: the cheap model says it needs an action → re-route
        # the ORIGINAL request (with its tools) to the Portal.
        if NEED_TOOL in text:
            log.info("1min requested tool → re-routing to portal")
            _record_backend("portal")
            _log_usage(user_text, action=True)  # learned: this "text" task is really an action
            return await _forward_portal(body, PORTAL_TOOL_MODEL)

        _record_backend("1min")
        _log_usage(user_text, action=False)  # clean text task answered by 1min
        return {
            "id": "chatcmpl-1min",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": route["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    # Heavy → Portal (full context + tools, cached prefix).
    _record_backend("portal")
    _log_usage(user_text, action=None)  # heavy task — not action/text learnable
    return await _forward_portal(body, route["model"])


async def _forward_portal(body: dict, model: str):
    """Forward the ORIGINAL request to the Portal proxy, preserving tools and
    tool_choice so the agent can actually perform actions."""
    payload = _portal_payload(body, model)
    stream = bool(payload.get("stream", False))

    if stream:
        try:
            async with httpx.AsyncClient(timeout=PORTAL_REQUEST_TIMEOUT) as client:
                upstream = await client.post(
                    f"{PORTAL_PROXY_URL}/chat/completions",
                    headers={"Authorization": "Bearer hermes-router"},
                    json=payload,
                )
        except Exception as exc:
            return _openai_error(502, "portal_proxy_error", f"portal proxy unreachable: {exc}")
        return StreamingResponse(
            _relay_stream(upstream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        data, source = await _get_portal_json(payload, model)
    except Exception as exc:
        return _openai_error(502, "portal_proxy_error", f"portal proxy unreachable: {exc}")
    return JSONResponse(content=data, headers={"X-Hermes-Router-Cache": source})


async def _relay_stream(upstream: httpx.Response):
    """Re-emit the portal proxy's SSE stream unchanged (with a clean [DONE])."""
    try:
        async for line in upstream.aiter_lines():
            if line:
                yield line + "\n\n"
    finally:
        yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ROUTER_PORT", "8400")))
