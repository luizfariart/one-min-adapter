# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
