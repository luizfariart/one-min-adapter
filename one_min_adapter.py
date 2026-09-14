"""1min.ai OpenAI-compatible adapter.

Exposes a local OpenAI-style ``/v1/chat/completions`` endpoint and translates
requests to the 1min.ai proprietary API (``POST /api/chat-with-ai``) and back.
The Hermes agent points a ``custom`` provider at this adapter, so the 1min.ai
credit balance becomes just another routing target for the model-router.

Key properties:

- **Graceful failure** — any 1min.ai error (401, 429, exhausted credits, network
  timeout) is translated to a proper OpenAI-style error response with the right
  HTTP status, so the caller's fallback chain can catch it and move on. The
  adapter never hangs and never returns a half-formed 200.
- **Streaming** — ``?isStreaming=true`` SSE from 1min.ai is re-emitted as
  OpenAI-style ``data: {...}`` SSE chunks ending in ``data: [DONE]``.
- **Model mapping** — OpenAI-style model names are mapped to 1min.ai model IDs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger("one_min_adapter")

API_BASE = "https://api.1min.ai"
CHAT_PATH = "/api/chat-with-ai"
DEFAULT_MODEL = "deepseek-flash"
REQUEST_TIMEOUT = 300  # seconds; 1min.ai can be slow for reasoning models

# OpenAI-style model name (what Hermes sends) → 1min.ai model id.
# Left side is intentionally short/stable; extend as the router needs more.
MODEL_MAP: dict[str, str] = {
    "deepseek-flash": "deepseek-flash",
    "deepseek-v4-flash": "deepseek-flash",
    "deepseek-chat": "deepseek-chat",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "deepseek-reasoner": "deepseek-reasoner",
    "gpt-5.4-mini": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-4o-mini",
    "gemini-3.5-flash": "gemini-3.5-flash",
    "gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "claude-4.5-haiku": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-4.6-sonnet": "us.anthropic.claude-sonnet-4-6",
    "grok-4": "grok-4-0709",
    "qwen-flash": "qwen-flash",
    "glm-5.1": "glm-5.1",
}


def _api_key() -> str:
    """Read the 1min.ai key from the secure file or env, never hardcoded."""
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


def _map_model(model: Optional[str]) -> str:
    if not model:
        return DEFAULT_MODEL
    return MODEL_MAP.get(model, model)  # pass through unknown ids unchanged


def _build_prompt(messages: list[dict]) -> str:
    """Flatten OpenAI messages into a single prompt string for 1min.ai.

    1min.ai's ``promptObject.prompt`` is a plain string (no separate system
    role). Join system + user + assistant turns; the system message (if any)
    goes first, framed so the model keeps it in mind.
    """
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
            parts.append(content)  # user (and any tool/function text)
    out = []
    if system:
        out.append("System instructions:\n" + "\n".join(system))
    out.extend(parts)
    return "\n\n".join(out)


def _openai_error(status: int, code: str, message: str) -> JSONResponse:
    """OpenAI-style error envelope so Hermes' fallback chain can handle it."""
    return JSONResponse(
        status_code=status,
        content={"error": {"type": code, "message": message}},
    )


def _extract_text(ai_record: dict) -> str:
    """Pull the response text out of a 1min.ai aiRecord."""
    detail = ai_record.get("aiRecordDetail") or {}
    result = detail.get("resultObject") or []
    if isinstance(result, list):
        return "".join(str(x) for x in result)
    return str(result)


def _call_1min(messages: list[dict], model: str, **kwargs) -> dict:
    """Non-streaming call. Raises on transport error; returns parsed JSON."""
    key = _api_key()
    if not key:
        raise ValueError("1min.ai API key not configured")
    payload: dict[str, Any] = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": _map_model(model),
        "promptObject": {
            "prompt": _build_prompt(messages),
            "settings": {
                "webSearchSettings": {"webSearch": False},
            },
        },
    }
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE}{CHAT_PATH}",
            headers={"API-KEY": key, "Content-Type": "application/json"},
            json=payload,
        )
    if resp.status_code == 401:
        raise PermissionError("1min.ai returned 401 — invalid/revoked API key")
    if resp.status_code == 429:
        raise RuntimeError("1min.ai returned 429 — rate limited or credits exhausted")
    resp.raise_for_status()
    data = resp.json()
    record = data.get("aiRecord") or {}
    if record.get("status") == "FAILED":
        raise RuntimeError("1min.ai request FAILED")
    return record


def _stream_1min(messages: list[dict], model: str):
    """Yield OpenAI-style SSE chunks from 1min.ai's streaming endpoint."""
    key = _api_key()
    if not key:
        yield "data: " + json.dumps({"error": {"message": "1min.ai API key not configured"}}) + "\n\n"
        yield "data: [DONE]\n\n"
        return
    payload: dict[str, Any] = {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": _map_model(model),
        "promptObject": {
            "prompt": _build_prompt(messages),
            "settings": {"webSearchSettings": {"webSearch": False}},
        },
    }
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            with client.stream(
                "POST",
                f"{API_BASE}{CHAT_PATH}?isStreaming=true",
                headers={"API-KEY": key, "Content-Type": "application/json"},
                json=payload,
            ) as resp:
                if resp.status_code == 401:
                    yield _sse_error("1min.ai returned 401 — invalid/revoked API key")
                    return
                if resp.status_code == 429:
                    yield _sse_error("1min.ai returned 429 — rate limited or credits exhausted")
                    return
                if resp.status_code >= 400:
                    yield _sse_error(f"1min.ai returned HTTP {resp.status_code}")
                    return
                # 1min.ai SSE: "event: content\ndata: {\"content\": ...}" etc.
                for raw in resp.iter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[len("data:"):].strip()
                    if data in ("[DONE]", ""):
                        continue
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    content = obj.get("content") if isinstance(obj, dict) else None
                    if content:
                        chunk = {
                            "id": "chatcmpl-1min",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": _map_model(model),
                            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(chunk) + "\n\n"
    except Exception as exc:  # network/timeout — must not kill the stream silently
        yield _sse_error(f"1min.ai adapter error: {exc}")
    finally:
        yield "data: [DONE]\n\n"


def _sse_error(message: str) -> str:
    obj = {"error": {"type": "one_min_error", "message": message}}
    return "data: " + json.dumps(obj) + "\n\n"


app = FastAPI(title="1min.ai OpenAI-compatible adapter")


@app.get("/health")
def health():
    key = _api_key()
    return {"ok": True, "key_configured": bool(key)}


@app.get("/v1/models")
def list_models():
    # Minimal catalog: the models this adapter exposes to Hermes.
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "owned_by": "1min-ai"} for mid in MODEL_MAP
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    model = body.get("model")
    stream = bool(body.get("stream", False))

    if stream:
        return StreamingResponse(
            _stream_1min(messages, model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        record = _call_1min(messages, model)
    except PermissionError as exc:
        return _openai_error(401, "authentication_error", str(exc))
    except RuntimeError as exc:
        # Rate limit OR credits exhausted — map to 429 so the fallback chain
        # knows this provider is unavailable right now.
        return _openai_error(429, "rate_limit_exceeded", str(exc))
    except httpx.HTTPStatusError as exc:
        return _openai_error(502, "upstream_error", f"1min.ai HTTP {exc.response.status_code}")
    except Exception as exc:
        return _openai_error(502, "adapter_error", str(exc))

    text = _extract_text(record)
    return {
        "id": "chatcmpl-1min",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _map_model(model),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ONEMIN_PORT", "8400")))
