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


# Free, unauthenticated sports-news RSS feeds scanned in addition to the per-team Google-News search.
# These are single feeds covering many teams (unlike the per-team Google query), so their headlines
# are matched against EVERY squad player and mapped back to the player's team. ESPN relays national-
# team withdrawal/availability news (federation + coach announcements), which is exactly the signal
# the results-only model lacks. Add more (Sky, BBC, Guardian) here — same parser, no keys, no cost.
_GENERAL_FEEDS: list[tuple[str, str]] = [
    ("ESPN", "https://www.espn.com/espn/rss/soccer/news"),
]


def _fetch_rss_titles(url: str, limit: int = 40, timeout: float = 8.0) -> list[str]:
    """Headline titles from a public RSS feed (network). Returns [] on any failure (offline/blocked)."""
    import urllib.request
    import xml.etree.ElementTree as ET

    from .sentiment import _ssl_context
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            root = ET.fromstring(resp.read())
    except Exception:
        return []
    return [(item.findtext("title") or "").strip() for item in list(root.iter("item"))[:limit]]


def general_feed_injury_flags(
    squads: dict | None = None, feeds: list[tuple[str, str]] | None = None, limit: int = 40
) -> dict[str, list[tuple[str, str]]]:
    """{team: [(player, '[Source] headline')]} from shared RSS feeds (e.g. ESPN).

    Each feed's headlines are scanned against *all* squad players (via ``scan_headlines``) and the
    flagged player is mapped back to their national team. Purely additive to the Google-News flags;
    output stays verify-only. Network-free logic lives in the pure parsers above."""
    squads = squads if squads is not None else squads_mod.load_squads()
    feeds = feeds if feeds is not None else _GENERAL_FEEDS
    player_team = {pl.name: team for team, sq in squads.items() for pl in sq.players}
    all_players = list(player_team)
    out: dict[str, list[tuple[str, str]]] = {}
    for source, url in feeds:
        for player, headline in scan_headlines(_fetch_rss_titles(url, limit=limit), all_players):
            out.setdefault(player_team[player], []).append((player, f"[{source}] {headline}"))
    return out


# ESPN's dedicated WC2026 injury-tracker article — a curated, per-player list ("Name, Country,
# Injury: type, description"). Richer than headlines but a single HTML page, so it is parsed from the
# visible TEXT (name-based), not the tag structure, to survive redesigns. If ESPN rotates the article
# id, update this URL; a dead URL simply reports 'fetch_failed' (non-fatal — see parse_tracker_text).
_ESPN_TRACKER_URL = (
    "https://www.espn.com/soccer/story/_/id/48572979/"
    "2026-fifa-world-cup-injuries-tracker-which-stars-miss-latest-info"
)
# Health strings that mean the scraper likely broke (page redesigned/blocked/JS-only or URL dead), as
# opposed to 'ok'. The suggester turns any non-'ok' into a loud warning; NONE of them can affect the
# forecast, which reads only the human-confirmed wc2026_injuries.txt.
TRACKER_DEGRADED = {"fetch_failed", "degraded_empty", "degraded_format", "degraded_overmatch", "parsed_zero"}


def _visible_text(html: str) -> str:
    """Tag-stripped visible text of an HTML page (scripts/styles removed, whitespace collapsed)."""
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def parse_tracker_text(text: str, player_team: dict[str, str]) -> tuple[dict[str, list[tuple[str, str]]], str]:
    """Parse ESPN-tracker visible text -> ({team: [(player, status)]}, health). Pure & network-free.

    Precision path: each 'Injury: <type>' is attributed to the squad player named in the short window
    just before it (nearest-name), giving a real status snippet. If that yields nothing on a page that
    clearly loaded, fall back to flagging any squad player named anywhere in the article (lower
    precision, health='degraded_format'). Health is 'ok' only on the precision path; every other value
    is a breakage signal the caller surfaces. Output is always verify-only.
    """
    all_players = list(player_team)
    if not text or len(text) < 800 or "Injury" not in text:
        return {}, "degraded_empty"  # JS shell / blocked / redesigned away
    teams = set(player_team.values())
    by_team: dict[str, list[str]] = {}
    for pl, tm in player_team.items():
        by_team.setdefault(tm, []).append(pl)

    # Each entry reads "<Name> , <Country> Injury: <type>". Anchor on that exact shape: the 1–4 word
    # tokens immediately before ", <Country> Injury:" are the subject, and the country must be a real
    # team AND the name must be one of THAT team's squad — a double constraint that rejects the player
    # names ESPN sprinkles through the prose (e.g. "handed Neymar a lifeline" before Gnabry's entry).
    entry = re.compile(r"([\w'’.\-]+(?:\s+[\w'’.\-]+){0,3})\s*,\s*([\w'’. \-]{3,30}?)\s+Injury:\s*([^.]{2,50})")
    out: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for m in entry.finditer(text):
        name_raw, country, typ = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        team = country if country in teams else next(
            (t for t in teams if country.endswith(t) or t.endswith(country)), None)
        if team is None:
            continue  # country label didn't resolve -> prose bleed, not a real entry
        hits = players_in_text(name_raw, by_team[team])  # must be a squad player of *this* team
        if not hits:
            continue
        player, key = hits[-1], (team, hits[-1])
        if key in seen:
            continue
        seen.add(key)
        out.setdefault(team, []).append((player, f"[ESPN tracker] Injury: {typ}"))
    if not out:  # page loaded but the "Name , Country Injury:" shape matched nothing -> format changed
        for player in players_in_text(text, all_players):
            out.setdefault(player_team[player], []).append(
                (player, "[ESPN tracker] named in injury tracker (verify)"))
        return out, ("degraded_format" if out else "parsed_zero")
    if sum(len(v) for v in out.values()) > 60:  # implausible for an injury list -> probable mis-parse
        return out, "degraded_overmatch"
    return out, "ok"


def espn_tracker_flags(
    squads: dict | None = None, url: str = _ESPN_TRACKER_URL, timeout: float = 15.0
) -> tuple[dict[str, list[tuple[str, str]]], str]:
    """Fetch + parse ESPN's injury-tracker article. Fail-soft: never raises, returns ({}, health)."""
    import urllib.request

    squads = squads if squads is not None else squads_mod.load_squads()
    player_team = {pl.name: team for team, sq in squads.items() for pl in sq.players}
    from .sentiment import _ssl_context
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            html = resp.read().decode("utf-8", "ignore")
    except Exception as e:
        return {}, f"fetch_failed:{type(e).__name__}"
    return parse_tracker_text(_visible_text(html), player_team)


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
    for team, fl in general_feed_injury_flags(squads).items():  # ESPN (+ any other shared feeds) RSS
        flags.setdefault(team, []).extend(fl)

    # Tier 2: ESPN's dedicated injury-tracker article (richer per-player status). Fully isolated — a
    # scraper break contributes zero flags and only raises a warning here; it can never reach the
    # forecast (which reads wc2026_injuries.txt) nor suppress the other sources above.
    tracker_flags, tracker_health = espn_tracker_flags(squads)
    for team, fl in tracker_flags.items():
        flags.setdefault(team, []).extend(fl)
    degraded = tracker_health.split(":")[0] in TRACKER_DEGRADED
    if degraded:
        import sys
        print(f"WARNING: ESPN injury-tracker scraper degraded ({tracker_health}); tracker suggestions "
              "may be incomplete. RSS + Google News + Wikipedia sources are unaffected, and the "
              "forecast is untouched (it reads only the confirmed wc2026_injuries.txt).", file=sys.stderr)

    replaced = wikipedia_replacements()

    lines = [
        "# AUTO-SUGGESTED availability — VERIFY before copying confirmed lines into wc2026_injuries.txt.",
        "# Sources: Google News RSS + ESPN soccer RSS + ESPN injury-tracker article + Wikipedia notes",
        "# (all noisy — a 'returns from injury' line can false-flag a fit player). Verify each line.",
        "# Format matches the live file: 'Team: Player1, Player2'.",
    ]
    lines.append(f"# ESPN injury-tracker scraper: {tracker_health}"
                 + ("  ⚠ DEGRADED — tracker rows may be missing (other sources unaffected)" if degraded else ""))
    lines.append("")
    for team, fl in sorted(flags.items()):
        seen_pairs = list(dict.fromkeys(fl))  # dedupe identical (player, headline) across feeds
        players = sorted({p for p, _ in seen_pairs})
        lines.append(f"{team}: {', '.join(players)}")
        for p, headline in seen_pairs:
            lines.append(f"#   {p} <- {headline[:90]}")
    if replaced:
        lines.append("")
        lines.append("# Wikipedia replacement notes (team unmapped — verify):")
        lines += [f"#   {p}" for p in replaced]

    p = Path(out_path) if out_path else _RAW / "wc2026_injuries.suggested.txt"
    p.write_text("\n".join(lines) + "\n")
    return p
