# 1min.ai Adapter

An OpenAI-compatible adapter for the [1min.ai](https://1min.ai) API. It exposes
a local `POST /v1/chat/completions` endpoint and translates requests to 1min.ai's
proprietary `POST /api/chat-with-ai` format — and responses back. Point any
OpenAI-compatible client (Hermes Agent, Codex, Aider, etc.) at it and use your
1min.ai credit balance as a model provider.

## Why this exists

1min.ai sells credit bundles (GPT, Claude, Gemini, DeepSeek, Grok, and more
through one key), but its API is **not** OpenAI-compatible:

| | OpenAI-compatible | 1min.ai native |
|---|---|---|
| Endpoint | `POST /v1/chat/completions` | `POST /api/chat-with-ai` |
| Auth header | `Authorization: Bearer …` | `API-KEY: …` |
| Request body | `{ messages: [...] }` | `{ type, model, promptObject: { prompt } }` |
| Response | `choices[0].message.content` | `aiRecord.aiRecordDetail.resultObject` |

This adapter sits in between and speaks OpenAI on the local side, 1min.ai on the
remote side. It also bridges the two streaming formats (SSE ↔ SSE).

## Features

- **Non-streaming** and **streaming** (`stream: true`, SSE) chat completions.
- **Model mapping** — short OpenAI-style names (`deepseek-v4-flash`) map to
  1min.ai model IDs (`deepseek-flash`); unknown IDs pass through unchanged.
- **Graceful failure** — 401, 429, exhausted credits, network timeouts, and any
  1min.ai error become proper OpenAI-style error responses (or an SSE `error`
  chunk + `[DONE]`), so the caller's fallback chain catches them and moves on.
  The adapter never hangs and never returns a half-formed 200.
- **`GET /v1/models`** and **`GET /health`** for discovery and monitoring.
- **Zero secrets in the repo** — the API key is read from a `chmod 600` file
  (`~/.hermes/secrets/1min.key` by default) or the `ONEMIN_API_KEY` env var.

## Install

```bash
# 1. Save your API key (never commit it)
printf '%s' '<your-key>' > ~/.hermes/secrets/1min.key
chmod 600 ~/.hermes/secrets/1min.key

# 2. Run (foreground) — or use install.sh for a launchd daemon
~/.hermes/hermes-agent/venv/bin/python3 one_min_adapter.py

# 3. Verify
curl -s http://127.0.0.1:8400/health
curl -s http://127.0.0.1:8400/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-flash","messages":[{"role":"user","content":"Say OK"}]}'
```

### Run as a launchd daemon (macOS)

```bash
./install.sh
```

Generates `~/Library/LaunchAgents/com.hermes.one-min-adapter.plist` with this
user's paths, registers it, and keeps it alive across reboots (KeepAlive).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ONEMIN_PORT` | `8400` | Local listen port |
| `ONEMIN_KEY_FILE` | `~/.hermes/secrets/1min.key` | Path to the API key file |
| `ONEMIN_API_KEY` | — | Inline key (overrides the file) |

## Using with Hermes Agent

Register the adapter as a `custom` provider, then route to it via a model alias:

```yaml
# config.yaml
model:
  aliases:
    fast-1min:
      model: deepseek-v4-flash
      provider: custom
      base_url: "http://127.0.0.1:8400/v1"
```

The model-router skill routes cheap `fast` tasks to this alias and heavy tasks
to the primary provider, so the credit balance absorbs the low-value work.

## Development

```bash
~/.hermes/hermes-agent/venv/bin/python -m pytest tests/ -q
```

## Limitations

- The 1min.ai chat endpoint takes a single prompt string, so multi-turn
  `messages` are flattened (system → framed, assistant → `Assistant:` prefix).
  Tool/function-calling is not yet supported; multimodal (image/video/audio)
  endpoints are not exposed by this adapter (1min.ai supports them natively).

## License

MIT — see [LICENSE](LICENSE).
