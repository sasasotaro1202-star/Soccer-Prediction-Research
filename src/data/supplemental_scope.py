from __future__ import annotations

"""Scope registry for supplemental competitions."""

SUPPLEMENTAL_COMPETITIONS = (
    "EMPERORS_CUP", "INTER_HIGH_SCHOOL", "U18_INTERNATIONAL",
    "U20_INTERNATIONAL", "U23_INTERNATIONAL", "FIFA_WORLD_CUP",
    "FIFA_WORLD_CUP_QUALIFIERS", "AFC_ASIAN_CUP", "AFC_ASIAN_CUP_QUALIFIERS",
    "UEFA_EURO", "UEFA_EURO_QUALIFIERS", "UEFA_NATIONS_LEAGUE",
    "INTERNATIONAL_FRIENDLIES",
)

SOURCE_CANDIDATES = {
    "EMPERORS_CUP": ("JFA", "openfootball"),
    "INTER_HIGH_SCHOOL": ("JFA",),
    "U18_INTERNATIONAL": ("JFA", "openfootball"),
    "U20_INTERNATIONAL": ("JFA", "openfootball"),
    "U23_INTERNATIONAL": ("JFA", "openfootball"),
    "FIFA_WORLD_CUP": ("openfootball", "FIFA"),
    "FIFA_WORLD_CUP_QUALIFIERS": ("openfootball", "FIFA"),
    "AFC_ASIAN_CUP": ("openfootball", "AFC"),
    "AFC_ASIAN_CUP_QUALIFIERS": ("openfootball", "AFC"),
    "UEFA_EURO": ("openfootball", "UEFA"),
    "UEFA_EURO_QUALIFIERS": ("openfootball", "UEFA"),
    "UEFA_NATIONS_LEAGUE": ("openfootball", "UEFA"),
    "INTERNATIONAL_FRIENDLIES": ("openfootball",),
}


def supplemental_scope() -> tuple[str, ...]:
    return SUPPLEMENTAL_COMPETITIONS
