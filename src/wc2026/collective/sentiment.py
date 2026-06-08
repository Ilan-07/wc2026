"""Sentiment / news-pulse feed (plan Tier B — but DISPLAY ONLY).

IMPORTANT: per the plan, this never enters the probabilistic forecast. Two findings already
showed extra layers don't beat the market, and sentiment is the noisiest signal of all. So it
lives purely in the UI: a glanceable news feed plus a heuristic "Pulse" so users can eyeball the
mood around a team. It is explicitly excluded from prediction to avoid hand-weighted indices
leaking into a calibrated model.

Source: Google News RSS search (no API key, returns headlines + source + date). Sentiment is a
transparent lexicon score over headlines — deliberately simple, since it is informational only.
"""

from __future__ import annotations

import ssl
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle; fall back to an unverified context (public RSS only)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()

_POS = {
    "win", "wins", "won", "beat", "boost", "star", "brilliant", "confident", "hope",
    "strong", "favourite", "favorite", "fit", "return", "returns", "top", "victory",
    "impressive", "clinch", "secure", "rise", "dominant", "magic", "hero",
}
_NEG = {
    "injury", "injured", "doubt", "crisis", "loss", "lose", "lost", "out", "fear",
    "blow", "concern", "concerns", "struggle", "struggles", "axed", "ban", "banned",
    "fail", "fails", "defeat", "sack", "sacked", "row", "exit", "worry", "ruled",
}


@dataclass
class NewsItem:
    title: str
    source: str
    date: str
    link: str
    score: int  # +1 per positive word, -1 per negative word in the headline


@dataclass
class TeamPulse:
    team: str
    items: list[NewsItem] = field(default_factory=list)
    pulse: float = 50.0  # 0-100, 50 = neutral
    mood: str = "neutral"


def _headline_score(title: str) -> int:
    words = {w.strip(".,!?:;\"'()").lower() for w in title.split()}
    return len(words & _POS) - len(words & _NEG)


def fetch_team_news(team: str, limit: int = 6, timeout: float = 8.0) -> TeamPulse:
    """Fetch recent headlines for a team and compute a glanceable pulse (0-100)."""
    q = urllib.parse.quote(f'"{team}" football world cup')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            root = ET.fromstring(resp.read())
    except Exception:
        return TeamPulse(team=team)  # offline / blocked -> empty feed, neutral pulse

    items: list[NewsItem] = []
    for item in list(root.iter("item"))[:limit]:
        title = (item.findtext("title") or "").strip()
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None else ""
        items.append(
            NewsItem(
                title=title,
                source=source,
                date=(item.findtext("pubDate") or "")[:16],
                link=(item.findtext("link") or "").strip(),
                score=_headline_score(title),
            )
        )

    if items:
        avg = sum(i.score for i in items) / len(items)
        pulse = max(0.0, min(100.0, 50.0 + 18.0 * avg))  # squash to 0-100
        mood = "positive" if pulse >= 60 else "negative" if pulse <= 40 else "neutral"
    else:
        pulse, mood = 50.0, "neutral"
    return TeamPulse(team=team, items=items, pulse=round(pulse, 1), mood=mood)


def fetch_many(teams: list[str], limit: int = 6) -> dict[str, TeamPulse]:
    """Fetch news for several teams (sequential; small N intended for the dashboard)."""
    return {t: fetch_team_news(t, limit=limit) for t in teams}
