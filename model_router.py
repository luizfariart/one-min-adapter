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

REQUEST_TIMEOUT = 15  # per-request cap: 300→60→15s; fallback to Portal kicks in quickly

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
    return _exemplars_state["centroids"]


def _needs_tool(text: str) -> bool:
    """True when the task vector is closer to the 'action' centroid than 'text'."""
    vec = _vec(text)
    if not vec:
        return False
    centroids = _get_centroids()
    return _cosine(vec, centroids["action"]) > _cosine(vec, centroids["text"])


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
    text = _last_user_text(messages).lower()
    if not text:
        return "chat"

    for cls, kws in _KEYWORD_RULES:
        if any(kw in text for kw in kws):
            return cls

    try:
        return _classify_with_model(text)
    except Exception:
        return "chat"


def _classify_with_model(text: str) -> str:
    """Free/cheap classification via 1min.ai deepseek-flash."""
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
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                f"{ONEMIN_API_BASE}{ONEMIN_CHAT_PATH}",
                headers={"API-KEY": key, "Content-Type": "application/json"},
                json=payload,
            )
        resp.raise_for_status()
    except Exception:
        return "chat"
    record = resp.json().get("aiRecord") or {}
    detail = record.get("aiRecordDetail") or {}
    result = detail.get("resultObject") or []
    answer = "".join(str(x) for x in result).strip().lower()
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
    key = _api_key()
    if not key:
        raise ValueError("1min.ai API key not configured")
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": _ONEMIN_MODEL_MAP.get(model, model),
        "promptObject": {"prompt": prompt, "settings": {"webSearchSettings": {"webSearch": False}}},
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
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


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #

app = FastAPI(title="Hermes model router")


@app.get("/health")
def health():
    return {
        "ok": True,
        "1min_key_configured": bool(_api_key()),
        "portal_proxy": PORTAL_PROXY_URL,
        "thrash": _detect_thrash(),
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
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            upstream = await client.post(
                f"{PORTAL_PROXY_URL}/chat/completions",
                headers={"Authorization": "Bearer hermes-router"},
                json=payload,
            )
    except Exception as exc:
        return _openai_error(502, "portal_proxy_error", f"portal proxy unreachable: {exc}")

    if stream:
        return StreamingResponse(
            _relay_stream(upstream),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        data = upstream.json()
    except Exception:
        return _openai_error(502, "portal_proxy_error", f"portal proxy HTTP {upstream.status_code}, non-JSON body")
    if upstream.status_code >= 400:
        return JSONResponse(status_code=upstream.status_code, content=data)
    return JSONResponse(content=data)


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
