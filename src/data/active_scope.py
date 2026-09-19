"""Authoritative active competition scope for the current production build.

Only these competitions are actively acquired/researched. Everything else in the
broader catalog is parked until this scope passes all production gates.
"""

ACTIVE_COMPETITIONS = (
    "EPL", "AG_M", "AG_W", "ERE", "LL", "SA", "BL1",
    "J1", "J2", "J3", "FL1", "UCL", "UEL", "U23_M", "U18_M",
)

ACTIVE_COMPETITION_SET = frozenset(ACTIVE_COMPETITIONS)


def is_active_competition(code: str) -> bool:
    return str(code).strip() in ACTIVE_COMPETITION_SET
