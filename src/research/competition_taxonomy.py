"""Broad football competition/stage taxonomy used for scope discovery.

This classifies the *kind of event* separately from the provider's tournament name.
Classification is descriptive and research-only; it never grants production eligibility.
"""
from __future__ import annotations

import re
import unicodedata


def _norm(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def classify_competition_kind(name: object, slug: object = "", *, gender: str = "", age_group: str = "") -> str:
    text = f"{_norm(name)} {_norm(slug)}"
    if any(x in text for x in ("friendly", "friendlies", "test match", "exhibition")):
        return "friendly"
    if any(x in text for x in ("super cup", "supercup", "supercopa", "community shield", "supercoppa", "trophee des champions", "supertaça", "super taca")):
        return "super_cup"
    if any(x in text for x in ("league cup", "efl cup", "carabao cup", "j league cup", "copa de la liga")):
        return "league_cup"
    if any(x in text for x in ("qualifier", "qualifying", "qualification", "play-in")):
        return "qualifier"
    if any(x in text for x in ("champions league", "europa league", "conference league", "libertadores", "sudamericana", "confederation cup", "champions cup")):
        return "continental_club"
    if any(x in text for x in ("world cup", "euro", "copa america", "asian cup", "africa cup", "gold cup", "nations league", "asian games", "olympic", "olympics")):
        return "international_tournament"
    if any(x in text for x in ("cup", "copa", "pokal", "fa cup", "emperor")):
        return "domestic_cup"
    if any(x in text for x in ("u23", "u21", "u20", "u19", "u18", "u17", "u16", "youth", "reserve", "academy")):
        return "youth"
    if gender == "women" or any(x in text for x in ("women", "female", "ladies")):
        return "women"
    if any(x in text for x in ("regions", "regional", "east asian", "eaff", "gulf")):
        return "regional"
    if any(x in text for x in ("invitational", "premier cup", "summer series")):
        return "invitational"
    if "league" in text:
        return "league"
    if any(x in text for x in ("school", "interhigh", "high school")):
        return "school"
    return "other"


def classify_stage(name: object = "", slug: object = "", round_name: object = "", phase: object = "") -> str:
    text = f"{_norm(name)} {_norm(slug)} {_norm(round_name)} {_norm(phase)}"
    if any(x in text for x in ("final", "finals")):
        return "final"
    if any(x in text for x in ("semi final", "semifinal", "semi")):
        return "semi_final"
    if any(x in text for x in ("quarter final", "quarterfinal", "quarter")):
        return "quarter_final"
    if any(x in text for x in ("round of 16", "last 16", "r16")):
        return "round_of_16"
    if any(x in text for x in ("round of 32", "last 32", "r32")):
        return "round_of_32"
    if any(x in text for x in ("playoff", "play off", "promotion playoff", "relegation playoff", "play in")):
        return "playoff"
    if any(x in text for x in ("third place", "3rd place")):
        return "third_place"
    if any(x in text for x in ("group", "league phase", "group stage")):
        return "group_stage"
    if any(x in text for x in ("qualifying", "qualification", "qualifier", "preliminary")):
        return "qualifier"
    if any(x in text for x in ("regular season", "league stage", "league")):
        return "regular_season"
    return "unknown"
