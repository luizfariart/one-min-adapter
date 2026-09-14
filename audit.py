#!/usr/bin/env python3
"""Weekly paid audit — re-judge learned exemplars to break self-reinforcement.

The daily self-improvement loop can self-reinforce: a mislabeled task gets
logged and promoted back into the index, feeding its own error. This weekly job
asks a paid Portal model to re-classify every exemplar; any disagreement
corrects the index. Runs at most once per week, offline from the request path.

Usage:
    python3 audit.py [--dry-run] [--exemplars PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx  # noqa: E402

from exemplars import (  # noqa: E402
    EXEMPLARS_PATH,
    build_audit_prompt,
    load_exemplars,
    mark_run,
    parse_audit_response,
    save_exemplars,
    should_run,
)

PORTAL_PROXY_URL = os.environ.get("PORTAL_PROXY_URL", "http://127.0.0.1:8645/v1")
AUDIT_MODEL = os.environ.get("AUDIT_MODEL", "deepseek/deepseek-v4-pro")
STATE_PATH = Path.home() / ".hermes" / "one-min-adapter" / ".last_audit"
INTERVAL = 7 * 86400  # weekly


def call_audit_model(prompt: str) -> str:
    payload = {
        "model": AUDIT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    with httpx.Client(timeout=300) as client:
        resp = client.post(
            f"{PORTAL_PROXY_URL}/chat/completions",
            headers={"Authorization": "Bearer hermes-router"},
            json=payload,
        )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exemplars", type=Path, default=EXEMPLARS_PATH)
    ap.add_argument("--force", action="store_true", help="ignore the weekly interval")
    args = ap.parse_args()

    if not args.force and not should_run(STATE_PATH, INTERVAL):
        print("audit: skipped (ran within the last 7 days)")
        return 0

    before = load_exemplars(args.exemplars)
    prompt = build_audit_prompt(before)
    reply = call_audit_model(prompt)
    after, changes = parse_audit_response(reply, before)

    if changes:
        print(f"audit: corrected {len(changes)} exemplar(s)")
        for task, frm, to in changes:
            print(f"  - {task!r}: {frm} -> {to}")
    else:
        print("audit: no disagreements")

    if args.dry_run:
        return 0

    save_exemplars(after, args.exemplars)
    mark_run(STATE_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
