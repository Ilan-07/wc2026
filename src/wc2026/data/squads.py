"""Real WC2026 squad data from Wikipedia (plan P4 player/coach data layer).

Source: the "2026 FIFA World Cup squads" article (CC-BY-SA, fetched via the MediaWiki API —
a ToS-clean alternative to scraping Transfermarkt/FBref). One fetch yields every team's 26-man
squad with, per player: position, name, age, **national-team caps** (experience), national
goals, and **club + club country** (the basis for shared-club chemistry and league-quality
proxies). Each team's **head coach** is captured too.

    curl -sSL -o data/raw/wc2026_squads.json \\
      "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=main&titles=2026%20FIFA%20World%20Cup%20squads&redirects=1"

This is intentionally a clean, attributable data source. Per-player *club form* (minutes, xG)
is a later, heavier ingestion (FBref/Understat) that must respect each site's terms; this
module already provides the squad backbone those richer features attach to.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
TOURNAMENT_START = date(2026, 6, 11)

# A single player template line, e.g.
# {{nat fs g player|no=|pos=GK|name=[[Matěj Kovář]]|age={{birth date and age2|2026|6|11|2000|5|17}}|caps=19|goals=0|club=[[PSV Eindhoven]]|clubnat=NED}}
_PLAYER_RE = re.compile(r"\{\{nat fs (?:g )?player\|(.+?)\}\}\s*$", re.MULTILINE)
_BIRTH_RE = re.compile(r"birth date and age2\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)")
_HEADER_RE = re.compile(r"^==+\s*(.+?)\s*==+\s*$", re.MULTILINE)
# "Coach:" may be followed by a {{flagicon|XXX}} before the [[name]].
_COACH_RE = re.compile(r"Coach\s*:\s*(?:\{\{[^}]*\}\}\s*)?\[\[([^\]|]+)")


@dataclass
class Player:
    name: str
    pos: str  # GK / DF / MF / FW
    caps: int
    goals: int
    age: float | None
    club: str
    club_nat: str  # 3-letter country code of the club's league (e.g. ENG, ESP)


@dataclass
class Squad:
    team: str
    coach: str | None = None
    players: list[Player] = field(default_factory=list)

    def by_pos(self, pos: str) -> list[Player]:
        return [p for p in self.players if p.pos == pos]


def _strip_link(value: str) -> str:
    """Turn '[[SK Slavia Prague|Slavia Prague]]' or '[[PSV Eindhoven]]' into a plain name."""
    m = re.search(r"\[\[([^\]]+)\]\]", value)
    text = m.group(1) if m else value
    return text.split("|")[-1].strip()


def _parse_fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    # split on top-level '|' but keep nested {{...}} (age template) and [[...]]
    # (wikilinks like [[Rodri (footballer, born 1996)|Rodri]]) intact
    brace = bracket = 0
    cur = ""
    parts = []
    for ch in body:
        if ch == "|" and brace == 0 and bracket == 0:
            parts.append(cur)
            cur = ""
            continue
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace = max(0, brace - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)
        cur += ch
    parts.append(cur)
    for part in parts:
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


def _age_from(value: str) -> float | None:
    m = _BIRTH_RE.search(value)
    if not m:
        return None
    ry, rm, rd, by, bm, bd = (int(x) for x in m.groups())
    ref, born = date(ry, rm, rd), date(by, bm, bd)
    return round((ref - born).days / 365.25, 1)


def _to_int(value: str) -> int:
    m = re.search(r"-?\d+", value or "")
    return int(m.group(0)) if m else 0


def load_squads(path: str | Path | None = None) -> dict[str, Squad]:
    """Parse the squads wikitext into {team: Squad}. Accepts the raw API JSON or .wiki text."""
    p = Path(path) if path is not None else _RAW / "wc2026_squads.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — fetch it (see module docstring).")
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
        page = next(iter(data["query"]["pages"].values()))
        text = page["revisions"][0]["slots"]["main"]["*"]
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # drop editor comments

    # Locate team sub-section headers (skip group headers like "Group A" and generic ones).
    headers = [(m.start(), m.group(1).strip()) for m in _HEADER_RE.finditer(text)]
    squads: dict[str, Squad] = {}
    for i, (start, title) in enumerate(headers):
        if title.lower().startswith("group ") or title in {"References", "Notes", "Statistics"}:
            continue
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        block = text[start:end]
        players = []
        for pm in _PLAYER_RE.finditer(block):
            f = _parse_fields(pm.group(1))
            name = _strip_link(f.get("name", ""))
            if not name:
                continue
            players.append(
                Player(
                    name=name,
                    pos=f.get("pos", "").upper()[:2],
                    caps=_to_int(f.get("caps", "0")),
                    goals=_to_int(f.get("goals", "0")),
                    age=_age_from(f.get("age", "")),
                    club=_strip_link(f.get("club", "")),
                    club_nat=f.get("clubnat", "").strip().upper(),
                )
            )
        if len(players) < 11:  # not a real squad block
            continue
        coach_m = _COACH_RE.search(block)
        squads[title] = Squad(
            team=title,
            coach=_strip_link(coach_m.group(1)) if coach_m else None,
            players=players,
        )
    return squads
