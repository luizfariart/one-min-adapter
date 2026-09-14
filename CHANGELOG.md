# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2026-09-14

### Added

- **1min.ai circuit breaker.** After `ONEMIN_FAILURE_THRESHOLD` (2) consecutive
  1min failures, the router stops calling 1min entirely for
  `ONEMIN_CIRCUIT_OPEN_S` (180s) and routes straight to the Portal — no more
  dead-time waiting on a stalled backend. A real success (or genuine circuit
  expiry) re-closes it.
- **Breaker persistence.** The failure count and `disabled_until` are written
  atomically (tmp+rename) to
  `~/.hermes/one-min-adapter/router_state.json` and reloaded at startup, so a
  daemon restart no longer re-pays the discovery cost of two dead requests.
- **Per-backend timeouts.** `ONEMIN_REQUEST_TIMEOUT` (5s) + `CLASSIFY_TIMEOUT`
  (3s) keep the 1min path snappy; `PORTAL_REQUEST_TIMEOUT` (30s) gives the
  Portal normal model latency without false 502s.
- **Local runtime caches.** `classify` and `_needs_tool` results are cached per
  normalized text (TTL 600s). Short low-stakes prompts (`ok`, `sim`, `segue`,
  `valeu`, …) classify directly to `fast` with no model round-trip, so they
  fall back to `deepseek-v4-flash` instead of `gpt-5.4` when 1min is down.
- **Strict short response cache + in-flight coalescing.** Non-streaming,
  tool-free Portal responses are cached by exact body+model hash (TTL 45s).
  Identical requests that arrive while the first is still in flight share one
  upstream call — reported via `X-Hermes-Router-Cache: miss | coalesced | hit`.
- `/health` now exposes `1min_circuit_open`, `1min_disabled_for_s`,
  `cache_entries` and `portal_inflight`.

### Fixed

- **Circuit probe cleared failures below threshold.** `_is_1min_circuit_open()`
  reset the failure counter on every probe, so the breaker could never reach
  the threshold and open. It now resets only on real success or genuine expiry.

### Diagnosis (root cause of the latency and cost)

The 1min.ai `/api/chat-with-ai` endpoint stopped answering (read timeouts, zero
successful `fast`/`mid` routes logged). Every lightweight request then paid the
full 15s timeout before falling back to the Portal — the source of the
"why do responses take so long" symptom (~17s per affected turn) and of
repeated spend on duplicate/retried requests.

## [1.5.0] - 2026-09-14

### Added

- **Weekly paid audit** (`audit.py`, launchd every Monday 04:45) — a paid
  Portal model re-judges every exemplar; disagreements correct the index to
  break the daily loop's self-reinforcement.
- **Catch-up scheduling** — `RunAtLoad` + `should_run()` interval guard so the
  daily and weekly jobs run as soon as the Mac is awake if it was off at the
  scheduled time, without ever double-running.

## [1.4.0] - 2026-09-14

### Added

- **Continuous self-improvement.** The router logs every request to
  `usage.jsonl`; a daily launchd job (`self_improve.py`) promotes recurring
  task texts into the action/text exemplar index, prunes stale ones, and the
  router hot-reloads the result on the next request. Cold-start guard prevents
  pruning until there is enough usage evidence; each class is capped at 60
  exemplars to stay low-latency.
- Extracted the exemplar store into `exemplars.py` (shared by router and job),
  with the `improve()` function fully unit-tested (promote/prune/cap/guard).

## [1.3.0] - 2026-09-14

### Fixed

- **Tool-call deadlock.** The router was dropping `tools`/`tool_choice` when
  forwarding to the Portal, so no task could ever call a function — the agent
  would hang waiting for an action. Now the full request body is forwarded.

### Added

- **`[[NEED_TOOL]]` handshake** — the 1min.ai prompt asks the cheap model to
  emit `[[NEED_TOOL]]` when the task needs an external action; the router then
  re-routes the original request to the Portal.
- **Vector index (TF-IDF + cosine)** — deterministic, dependency-free
  "euclidean index" that classifies tasks as action vs text *before* the 1min
  call, so tool-requiring tasks skip the cheap round-trip entirely.

## [1.2.0] - 2026-09-14

### Added

- `mid` task class — simple explanations/questions now route to the cheap
  1min.ai backend (deepseek-chat) instead of the expensive Portal claude-opus.
- **Cost isolation** — `fast`/`mid` tasks are sent with an isolated prompt
  (task only, no system prompt/history), the "fresh chat" optimization.
- **Thrash detection** — rolling backend history; when the conversation bounces
  between Portal and 1min.ai, lightweight tasks switch to minimal isolation to
  protect the Portal cache and keep credit spend at its floor.

## [1.1.0] - 2026-09-14

### Added

- `model_router.py` — dynamic per-request routing: classifies every request
  (deterministic keyword rules → free 1min.ai model when ambiguous) and routes
  to 1min.ai (fast) or the Nous Portal (code/research/chat/review).
- `com.hermes.portal-proxy.plist` + install flow for the native `hermes proxy`
  OAuth bridge to the Portal (`:8645`).
- Graceful fallback: 1min failure → Portal; router down → Hermes `fallback_model`.

### Changed

- The production daemon now runs the router instead of the plain adapter.

## [1.0.0] - 2026-09-14

### Added

- OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming).
- Translation layer to/from 1min.ai's `/api/chat-with-ai` format.
- Model mapping table (DeepSeek, GPT, Gemini, Claude, Grok, Qwen, GLM).
- Graceful failure: 401/429/credits/timeouts → OpenAI-style errors, never a hang.
- `GET /v1/models` and `GET /health` endpoints.
- launchd daemon plist + `install.sh`.
- Unit tests for mapping, prompt flattening, and error envelopes.
