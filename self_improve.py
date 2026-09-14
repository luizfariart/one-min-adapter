#!/usr/bin/env python3
"""Daily self-improvement job for the model router.

Reads the usage log the router appends to on every request, learns recurring
task texts into exemplars, prunes stale ones, and saves the improved index.
Runs offline (launchd, daily) so the request hot path is never affected.

Usage:
    python3 self_improve.py [--dry-run] [--usage PATH] [--exemplars PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from exemplars import (  # noqa: E402
    EXEMPLARS_PATH,
    USAGE_LOG_PATH,
    WINDOW_DAYS,
    load_exemplars,
    improve,
    save_exemplars,
)


def load_usage(path: Path = USAGE_LOG_PATH) -> list[dict]:
    """Read JSONL usage log. Skips malformed lines; never raises on a bad file."""
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except Exception:
        return []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="print changes, do not write")
    ap.add_argument("--usage", type=Path, default=USAGE_LOG_PATH)
    ap.add_argument("--exemplars", type=Path, default=EXEMPLARS_PATH)
    args = ap.parse_args()

    usage = load_usage(args.usage)
    before = load_exemplars(args.exemplars)
    after = improve(before, usage, now=time.time())

    added_action = len(after["action"]) - len(before["action"])
    added_text = len(after["text"]) - len(before["text"])
    removed = (len(before["action"]) + len(before["text"])) - (len(after["action"]) + len(after["text"]))

    print(
        f"self-improve: usage={len(usage)} entries | "
        f"action {len(before['action'])}→{len(after['action'])} ({added_action:+d}) | "
        f"text {len(before['text'])}→{len(after['text'])} ({added_text:+d}) | "
        f"removed={removed}"
    )

    if args.dry_run:
        return 0

    save_exemplars(after, args.exemplars)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
