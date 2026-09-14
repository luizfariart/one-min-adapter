# 1min.ai Adapter + Model Router

Two local services that give the [Hermes Agent](https://hermes-agent.nousresearch.com)
dynamic, per-request model routing over the [1min.ai](https://1min.ai) credit
balance, without spending Portal dollars on cheap work.

- **`model_router.py`** — a local OpenAI-compatible proxy that classifies every
  request (deterministic rules → free/cheap model) and routes it to 1min.ai or
  the Nous Portal.
- **`one_min_adapter.py`** — a standalone 1min.ai adapter (chat only) for
  clients that just want a 1min.ai endpoint, no routing.
- **`com.hermes.portal-proxy.plist`** — launchd unit that runs `hermes proxy`
  (the native OAuth bridge to the Portal) on `:8645`.

## Architecture

```
Hermes Agent
    │  model.provider = custom
    │  base_url = http://127.0.0.1:8400/v1
    ▼
model_router (:8400)            ← classifies every request
    ├── fast → 1min.ai          ← pre-paid credit balance (deepseek-flash)
    └── code/research/chat/review
              │  forwards to
              ▼
        hermes proxy (:8645)    ← attaches Nous OAuth
              ▼
        Nous Portal
```

Every request is classified (keyword rules first — free; a cheap model only when
ambiguous) and routed. If 1min.ai fails (credits exhausted, rate limit, timeout),
the request falls back to the Portal automatically. If the router itself is
down, Hermes' `fallback_model` points straight at the Portal. **A request never
dies because a credit backend is down.**

## Why a router and not `/model` switching

Hermes has no native dynamic router: `/model` is manual, the fallback chain
fires on *failure* (not on classification), `smart_model_routing` is a
setup-wizard stub, and the `pre_llm_call` hook can inject context but cannot
change the model. The only correct way to route by task type on every turn is a
proxy in front of the agent — which is exactly what `model_router.py` is.

## Install

```bash
# 1. Save your 1min.ai API key (never commit it)
printf '%s' '<your-key>' > ~/.hermes/secrets/1min.key
chmod 600 ~/.hermes/secrets/1min.key

# 2. Install both launchd daemons (router + portal proxy)
./install.sh

# 3. Point Hermes at the router
hermes config set model.provider custom
hermes config set model.base_url "http://127.0.0.1:8400/v1"
hermes config set fallback_model.provider nous
hermes config set fallback_model.model "deepseek/deepseek-v4-pro"

# 4. Verify
curl -s http://127.0.0.1:8400/health
```

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ROUTER_PORT` | `8400` | Router listen port |
| `PORTAL_PROXY_URL` | `http://127.0.0.1:8645/v1` | Where the Portal OAuth proxy lives |
| `ONEMIN_KEY_FILE` | `~/.hermes/secrets/1min.key` | 1min.ai API key file |
| `ONEMIN_API_KEY` | — | Inline key (overrides the file) |

## Routing table

Edit `ROUTE_TABLE` in `model_router.py` to change which task class goes where:

| Class | Backend | Model |
|---|---|---|
| `fast` | 1min.ai | `deepseek-flash` (isolated prompt; fallback: Portal `deepseek-v4-flash`) |
| `mid` | 1min.ai | `deepseek-chat` (isolated prompt; fallback: Portal `deepseek-v4-flash`) |
| `code` | Portal | `deepseek/deepseek-v4-pro` |
| `research` | Portal | `anthropic/claude-opus-5` |
| `chat` | Portal | `openai/gpt-5.4` |
| `review` | Portal | `kwaipilot/kat-coder-pro-v2.5` |

## Cost isolation

Lightweight tasks (`fast`/`mid`) are sent to 1min.ai with an **isolated
prompt** — only the task, never the agent's large system prompt or history. A
formatting or explanation task does not need the agent's persona instructions,
so those credits stay unspent. This is the "open the light task in a fresh
chat" optimization.

Short follow-ups (≤15 words) keep the previous assistant reply as context so
"and in English?" still refers to something.

## Tool calling & the `[[NEED_TOOL]]` handshake

The router is tool-call transparent. It forwards the **full original request
body** (tools, tool_choice, temperature, stream) to the Portal, so tasks that
require an action can actually call functions.

Two layers decide when a task needs a tool:

1. **Vector index (pre-1min).** A deterministic TF-IDF + cosine index compares
   the task against "action" and "text" centroids. Tasks closer to *action*
   (web search, file/command/email/calendar, real-time data) skip the 1min
   round-trip and go straight to the Portal.
2. **`[[NEED_TOOL]]` handshake (in-1min).** The 1min.ai prompt tells the cheap
   model to reply with exactly `[[NEED_TOOL]]` if the task requires an external
   action it cannot do from text alone. The router detects the marker and
   re-routes the original request to the Portal.

Either way, a lightweight task that unexpectedly needs a tool is never left
hanging.

## Thrash detection

The router keeps a rolling history of the backend used per request. When the
conversation bounces between Portal and 1min.ai frequently (≥50% switches in the
last 8 requests), lightweight tasks switch to **minimal** isolation — dropping
even the follow-up context — so the credit spend stays at its floor while heavy
turns keep their cached Portal prefix.

## Development

```bash
~/.hermes/hermes-agent/venv/bin/python -m pytest tests/ -q
```

## Limitations

- 1min.ai's chat endpoint takes a single prompt string, so multi-turn `messages`
  are flattened. Tool/function-calling is not supported through the adapter.
- The standalone adapter covers chat only; 1min.ai's image/video/audio endpoints
  are not exposed yet (the router does not route them).
- Prompt caching on the Portal is broken on each model switch — the inherent
  cost of per-request dynamic routing.

## License

MIT — see [LICENSE](LICENSE).
