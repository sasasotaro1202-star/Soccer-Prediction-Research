"""Detect whether staged experience outputs contain a semantic change.

Generated status timestamps are operational noise and should not create repository
commits when the underlying matured-experience evidence is unchanged.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

VOLATILE_JSON_KEYS = frozenset({"generated_at_utc"})


def _run_git(*args: str) -> bytes | None:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode == 0:
        return proc.stdout
    if proc.returncode == 128 and (
        b"does not exist in" in proc.stderr
        or (b"pathspec" in proc.stderr and b"did not match any file" in proc.stderr)
    ):
        return None
    raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip() or "git command failed")


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in VOLATILE_JSON_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def semantic_json_equal(left: str, right: str) -> bool:
    try:
        left_value = _strip_volatile(json.loads(left))
        right_value = _strip_volatile(json.loads(right))
    except (TypeError, ValueError):
        return left == right
    return left_value == right_value


def staged_semantic_change(path: str) -> bool:
    staged = _run_git("show", f":{path}")
    if staged is None:
        # The workflow stages only files that actually exist. An absent optional
        # output therefore represents no change, not a guard failure.
        return False
    head = _run_git("show", f"HEAD:{path}")
    if head is None:
        return True

    staged_text = staged.decode("utf-8")
    head_text = head.decode("utf-8")
    if Path(path).suffix.lower() == ".json":
        return not semantic_json_equal(head_text, staged_text)
    return head != staged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)

    try:
        semantic = any(staged_semantic_change(path) for path in args.paths)
    except Exception as exc:
        print(f"experience semantic-change guard failed: {exc}", file=sys.stderr)
        return 2

    print("true" if semantic else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())