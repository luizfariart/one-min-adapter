# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-14

### Added

- OpenAI-compatible `/v1/chat/completions` endpoint (streaming + non-streaming).
- Translation layer to/from 1min.ai's `/api/chat-with-ai` format.
- Model mapping table (DeepSeek, GPT, Gemini, Claude, Grok, Qwen, GLM).
- Graceful failure: 401/429/credits/timeouts → OpenAI-style errors, never a hang.
- `GET /v1/models` and `GET /health` endpoints.
- launchd daemon plist + `install.sh`.
- Unit tests for mapping, prompt flattening, and error envelopes.
