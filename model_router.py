"""Model router — dynamic per-request routing with free classification.

A local OpenAI-compatible proxy that sits in front of the Hermes agent. On
every `/v1/chat/completions` request it:

1. **Classifies** the task from the conversation (deterministic keyword rules
   first, then a free/cheap model — 1min.ai deepseek-flash — only when the
   rules are inconclusive).
2. **Routes** the request to the right backend:
   - ``fast`` / trivial → 1min.ai directly (pre-paid credit balance, no Portal spend)
   - ``code`` / ``research`` / ``chat`` / ``review`` → the Hermes portal proxy
     (``http://127.0.0.1:8645/v1``), which attaches the Nous OAuth credentials.
3. **Falls back** cleanly: if the primary backend fails (credits exhausted,
   rate limit, timeout), it routes to the other backend. It never hangs and
   never returns a half-formed 200.

This replaces the manual ``/model <alias>`` switching: routing happens in the
proxy on every request, independent of whether the agent's own model remembers
to re-classify.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("model_router")

# Backends
ONEMIN_API_BASE = "https://api.1min.ai"
ONEMIN_CHAT_PATH = "/api/chat-with-ai"
PORTAL_PROXY_URL = os.environ.get("PORTAL_PROXY_URL", "http://127.0.0.1:8645/v1")

REQUEST_TIMEOUT = 300

# Task classes and their target backend + model.
# "fast" goes to 1min.ai (credit balance); the rest go to the Portal.
ROUTE_TABLE: dict[str, dict] = {
    "fast":     {"backend": "1min",  "model": "deepseek-flash",        "fallback_model": "deepseek/deepseek-v4-flash"},
    "code":     {"backend": "portal", "model": "deepseek/deepseek-v4-pro"},
    "research": {"backend": "portal", "model": "anthropic/claude-opus-5"},
    "chat":     {"backend": "portal", "model": "openai/gpt-5.4"},
    "review":   {"backend": "portal", "model": "kwaipilot/kat-coder-pro-v2.5"},
}

# Keyword rules → task class. Order matters (first match wins).
# Deterministic and free — most requests never touch the classifier model.
_KEYWORD_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("code", (
        "code", "codigo", "código", "debug", "debugar", "script", "função", "funcao",
        "bug", "refactor", "refatorar", "teste", "test", "api", "classe", "class",
        "implementar", "implement", "programa", "programar", "python", "typescript",
        "javascript", "sql", "regex", "compilar", "build", "commit", "git",
    )),
    ("research", (
        "pesquis", "research", "analis", "compare", "compar", "sintetiz", "fonte",
        "referência", "referencia", "estud", "artigo", "paper", "academic", "busca",
        "search", "explique", "explic", "por que", "porque", "como funciona",
    )),
    ("review", (
        "review", "revisar", "revisão", "revisao", "auditar", "audit", "validar",
        "validate", "code review", "inspecionar", "inspect",
    )),
    ("fast", (
        "formata", "formatar", "formate", "tabela", "lista",
        "traduz", "traduza", "resum", "classif", "uma palavra",
        "corrigir ortografia", "upper", "lower", "trim", "ordenar", "orden",
    )),
]


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
            # content as list of parts (multimodal) — flatten text
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


def classify(messages: list[dict]) -> str:
    """Classify the task. Deterministic rules first (free), then the cheap
    1min.ai model only if inconclusive. Never raises; falls back to 'chat'."""
    text = _last_user_text(messages).lower()
    if not text:
        return "chat"

    # Deterministic keyword pass (free, instant).
    for cls, kws in _KEYWORD_RULES:
        if any(kw in text for kw in kws):
            return cls

    # Inconclusive → ask the cheap model.
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
        "review, or fast. Reply with only that single word.\n\n"
        f"Task: {text[:500]}"
    )
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": "deepseek-flash",
        "promptObject": {"prompt": prompt},
    }
    with httpx.Client(timeout=60) as client:
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
    for cls in ROUTE_TABLE:
        if cls in answer:
            return cls
    return "chat"


# --------------------------------------------------------------------------- #
# 1min.ai backend (existing adapter logic, kept local to the router)
# --------------------------------------------------------------------------- #

_ONEMIN_MODEL_MAP = {
    "deepseek-flash": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-chat": "deepseek-chat",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
}


def _build_prompt(messages: list[dict]) -> str:
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


def _openai_error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": code, "message": message}})


def _call_1min(messages: list[dict], model: str) -> dict:
    key = _api_key()
    if not key:
        raise ValueError("1min.ai API key not configured")
    payload = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": _ONEMIN_MODEL_MAP.get(model, model),
        "promptObject": {"prompt": _build_prompt(messages), "settings": {"webSearchSettings": {"webSearch": False}}},
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


def _call_portal(messages: list[dict], model: str, stream: bool):
    """Forward to the Hermes portal proxy (attaches Nous OAuth)."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    return httpx.Client(timeout=REQUEST_TIMEOUT, headers={"Authorization": "Bearer hermes-router"})


def _extract_text(record: dict) -> str:
    detail = record.get("aiRecordDetail") or {}
    result = detail.get("resultObject") or []
    if isinstance(result, list):
        return "".join(str(x) for x in result)
    return str(result)


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
    task = classify(messages)
    route = ROUTE_TABLE.get(task, ROUTE_TABLE["chat"])
    log.info("routing -> %s (backend=%s)", task, route["backend"])

    # Fast → 1min.ai directly. ANY 1min failure (missing key, 401, 429, credits
    # exhausted, network) falls back to the Portal proxy so the request never
    # dies just because the credit backend is down.
    if route["backend"] == "1min":
        try:
            record = _call_1min(messages, route["model"])
        except Exception as exc:
            log.warning("1min backend failed (%s) — falling back to portal", exc)
            # Fall back to the cheap Portal flash model, not the expensive one.
            route = {"backend": "portal", "model": route.get("fallback_model") or "deepseek/deepseek-v4-flash"}
        else:
            text = _extract_text(record)
            return {
                "id": "chatcmpl-1min",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": route["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

    # Portal (or 1min fallback landed here) → forward to the portal proxy.
    return await _forward_portal(messages, route["model"], stream)


async def _forward_portal(messages: list[dict], model: str, stream: bool):
    payload = {"model": model, "messages": messages, "stream": stream}
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
