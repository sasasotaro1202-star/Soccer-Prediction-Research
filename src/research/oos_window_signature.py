from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Any


def exact_oos_window_signature(rows: Iterable[Mapping[str, Any]]) -> str | None:
    """Fingerprint exact chronological OOS windows, not merely fold IDs.

    Every window must expose a stable fold identifier plus explicit start/end
    boundaries. Missing boundaries fail closed.
    """
    identities: set[tuple[int, str, str]] = set()
    fold_ids: set[int] = set()
    seen = False
    for row in rows:
        seen = True
        try:
            fold_value = float(row.get("fold"))
        except (TypeError, ValueError):
            return None
        if not fold_value.is_integer():
            return None
        start = str(row.get("test_start") or row.get("oos_start") or "").strip()
        end = str(row.get("test_end") or row.get("oos_end") or "").strip()
        if not start or not end:
            return None
        fold_id = int(fold_value)
        if fold_id in fold_ids:
            # One exact OOS window is allowed per fold identity. Silent set-based
            # canonicalization would hide duplicate/multi-window folds and break
            # the reproducibility contract.
            return None
        fold_ids.add(fold_id)
        identities.add((fold_id, start, end))

    if not seen or not identities:
        return None

    canonical = "|".join(
        f"{fold}:{start}:{end}"
        for fold, start, end in sorted(identities)
    )
    return "oos:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
