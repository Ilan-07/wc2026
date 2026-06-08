"""Canonical team-name registry + coverage audit (gap #20).

Different sources name teams differently (results.csv, The Odds API, StatsBomb, Wikipedia). A
silent mismatch drops a team from the forecast without warning. This centralizes a canonical name
(the results.csv spelling) + known aliases, and provides an audit that *fails loudly* when a needed
team can't be matched — so coverage problems surface instead of corrupting the prediction.
"""

from __future__ import annotations

# variant spelling -> canonical (results.csv) name. Extend as new sources are added.
ALIASES: dict[str, str] = {
    "USA": "United States",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
    "DR Congo": "DR Congo",
    "Curacao": "Curaçao",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Turkiye": "Turkey",
}


def canonical(name: str) -> str:
    """Map a source spelling to the canonical team name (identity if already canonical)."""
    return ALIASES.get(name.strip(), name.strip())


def audit_coverage(needed: set[str], available: set[str], *, strict: bool = False) -> set[str]:
    """Return the needed teams missing from ``available`` (after canonicalization).

    With ``strict=True`` raises if any are missing — use before a forecast so a coverage gap
    surfaces loudly rather than silently dropping a team.
    """
    avail = {canonical(a) for a in available}
    missing = {t for t in (canonical(n) for n in needed) if t not in avail}
    if missing and strict:
        raise ValueError(f"team coverage gap — not found in source: {sorted(missing)}")
    return missing
