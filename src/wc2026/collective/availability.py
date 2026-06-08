"""Free player-availability suggestions (Gap 1) — Wikipedia replacements + news-RSS injury flags.

The biggest model-vs-market gap is availability: the market knows who's out, the results-only model
doesn't. Paid injury APIs are season-locked on the free tier, so this closes the gap the free, ToS-
clean way and writes **suggestions only** — you confirm them into the live ``wc2026_injuries.txt``:

  * **Wikipedia replacements** — the "2026 FIFA World Cup squads" article (already fetched via the
    MediaWiki API) records injury withdrawals/replacements; parse those notes.
  * **News-RSS flags** — reuse ``sentiment``'s free Google-News feed; flag headlines that pair an
    injury keyword with a squad player's surname.

Both are noisy → output goes to ``wc2026_injuries.suggested.txt`` (not the live file), and any signal
must clear a backtest before it's trusted. The pure parsers below are network-free and unit-tested;
the fetchers are thin wrappers over them.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..data import squads as squads_mod
from . import sentiment

_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"

# A headline flags availability if it has a strong-absence phrase on its own, or pairs an "out"
# cue with an injury word.
_STRONG = re.compile(r"\b(ruled out|sidelined|withdrew|withdrawn|suspend(?:ed)?|ban(?:ned)?)\b", re.I)
_OUT_CUE = re.compile(r"\b(out|ruled out|sidelined|withdraw|withdrew|miss(?:es|ing)?|doubt(?:ful)?)\b", re.I)
_INJURY_CUE = re.compile(r"\b(injur(?:y|ed|ies)|knock|strain|hamstring|knee|ankle|calf)\b", re.I)
# Wikipedia replacement note, e.g. "[[Gavi]] withdrew through injury and was replaced by [[Aleix]]".
_REPLACE_RE = re.compile(
    r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\][^.\n]{0,90}?(?:withdrew|withdrawn|ruled out|replaced)",
    re.I,
)


def is_injury_headline(title: str) -> bool:
    """True if a headline signals an absence: a strong-absence phrase, or out-cue + injury word."""
    return bool(_STRONG.search(title) or (_OUT_CUE.search(title) and _INJURY_CUE.search(title)))


def players_in_text(text: str, players: list[str]) -> list[str]:
    """Squad players any of whose name tokens (len ≥ 4) appears as a whole word in ``text``.

    Matching on *any* significant token (not just the surname) catches players widely known by a
    single name (Pedri, Rodri, Yamal) as well as by surname. It is intentionally loose — these are
    *suggestions* a human confirms — so a common shared token may over-flag; that's acceptable.
    """
    low = text.lower()
    hits = []
    for p in players:
        toks = [t for t in re.split(r"\s+", p) if len(t) >= 4]
        if any(re.search(rf"\b{re.escape(t.lower())}\b", low) for t in toks):
            hits.append(p)
    return hits


def scan_headlines(titles: list[str], players: list[str]) -> list[tuple[str, str]]:
    """(player, headline) pairs where an injury headline names a squad player. Pure/testable."""
    flags: list[tuple[str, str]] = []
    for t in titles:
        if is_injury_headline(t):
            for p in players_in_text(t, players):
                flags.append((p, t))
    return flags


# Org/citation link-text the replacement regex can catch near "replaced/withdrew" — not players.
_NON_PERSON = re.compile(
    r"\b(Association|Federation|Confederation|DAZN|ESPN|BBC|Sky|Athletic|League|Cup|DFB|UEFA|"
    r"FIFA|CONMEBOL|CONCACAF|News|Sport|Times|Guardian|Reuters|FC|United|City)\b",
    re.I,
)


def parse_replacements(wikitext: str) -> list[str]:
    """Players named as withdrawn/replaced in the squads article (deduped, order-preserving).

    Filters obvious non-players (federations, broadcasters, citation sources) and strips Wikipedia
    disambiguators like '(footballer)'. Still best-effort and team-unmapped — output is verify-only.
    """
    seen, out = set(), []
    for m in _REPLACE_RE.finditer(wikitext):
        name = re.sub(r"\s*\([^)]*\)", "", m.group(1)).strip()  # drop "(footballer)" etc.
        if not name or name in seen or _NON_PERSON.search(name):
            continue
        seen.add(name)
        out.append(name)
    return out


# ---------------------------------------------------------------- network/file wrappers
def _squads_wikitext(path: str | Path | None = None) -> str:
    import json

    p = Path(path) if path is not None else _RAW / "wc2026_squads.json"
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        data = json.loads(text)
        page = next(iter(data["query"]["pages"].values()))
        text = page["revisions"][0]["slots"]["main"]["*"]
    return text


def wikipedia_replacements(path: str | Path | None = None) -> list[str]:
    """Withdrawn/replaced players from the cached squads wikitext (best-effort, team-unmapped)."""
    return parse_replacements(_squads_wikitext(path))


def news_injury_flags(
    squads: dict | None = None, teams: list[str] | None = None, limit: int = 8
) -> dict[str, list[tuple[str, str]]]:
    """{team: [(player, headline)]} from the free Google-News feed (network)."""
    squads = squads if squads is not None else squads_mod.load_squads()
    teams = teams or list(squads)
    out: dict[str, list[tuple[str, str]]] = {}
    for team in teams:
        sq = squads.get(team)
        if sq is None:
            continue
        names = [pl.name for pl in sq.players]
        pulse = sentiment.fetch_team_news(team, limit=limit)
        flags = scan_headlines([i.title for i in pulse.items], names)
        if flags:
            out[team] = flags
    return out


def suggest_injuries(
    teams: list[str] | None = None, out_path: str | Path | None = None, limit: int = 8
) -> Path:
    """Write ``wc2026_injuries.suggested.txt`` (suggestions only — confirm into the live file)."""
    squads = squads_mod.load_squads()
    flags = news_injury_flags(squads, teams=teams, limit=limit)
    replaced = wikipedia_replacements()

    lines = [
        "# AUTO-SUGGESTED availability — VERIFY before copying confirmed lines into wc2026_injuries.txt.",
        "# Sources: Google News RSS injury headlines + Wikipedia replacement notes (both noisy).",
        "# Format matches the live file: 'Team: Player1, Player2'.",
        "",
    ]
    for team, fl in sorted(flags.items()):
        players = sorted({p for p, _ in fl})
        lines.append(f"{team}: {', '.join(players)}")
        for p, headline in fl:
            lines.append(f"#   {p} <- {headline[:90]}")
    if replaced:
        lines.append("")
        lines.append("# Wikipedia replacement notes (team unmapped — verify):")
        lines += [f"#   {p}" for p in replaced]

    p = Path(out_path) if out_path else _RAW / "wc2026_injuries.suggested.txt"
    p.write_text("\n".join(lines) + "\n")
    return p
