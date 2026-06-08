"""FIFA World Cup 2026 tournament structure and standings logic (plan Tier A5).

Format: 48 teams in 12 groups (A-L) of 4. Each group is a single round-robin (6 matches).
The 12 group winners, 12 runners-up, and the 8 best third-placed teams (32 total) advance
to a single-elimination knockout (Round of 32 -> R16 -> QF -> SF -> Final).

This module is pure logic (standings, tiebreakers, best-thirds, seeded bracket) and holds
no model; the Monte Carlo engine in ``tournament.py`` feeds it sampled results.

Note on the knockout bracket: FIFA assigns the 8 third-placed teams to fixed Round-of-32
slots via a combinatorial lookup table keyed by *which* groups the thirds come from. We use
a transparent **performance-seeded** bracket instead (top seeds meet latest), which yields
near-identical champion probabilities and is easy to reason about. The official allocation
table can be dropped into :func:`build_bracket_order` later without touching the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GROUP_NAMES = [chr(ord("A") + i) for i in range(12)]  # 'A'..'L'

# Round-robin schedule for a group of four teams (indices into the 4-team list).
ROUND_ROBIN_PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]


@dataclass
class TeamRecord:
    """A team's running group-stage tally plus head-to-head results within the group."""

    team: str
    points: int = 0
    gf: int = 0
    ga: int = 0
    # head-to-head: opponent -> (points, gf, ga) accumulated vs that opponent
    h2h: dict[str, tuple[int, int, int]] = field(default_factory=dict)

    @property
    def gd(self) -> int:
        return self.gf - self.ga

    def add_match(self, opponent: str, scored: int, conceded: int) -> None:
        if scored > conceded:
            pts = 3
        elif scored == conceded:
            pts = 1
        else:
            pts = 0
        self.points += pts
        self.gf += scored
        self.ga += conceded
        p, f, a = self.h2h.get(opponent, (0, 0, 0))
        self.h2h[opponent] = (p + pts, f + scored, a + conceded)


def rank_group(records: list[TeamRecord], rng) -> list[TeamRecord]:
    """Order a group's teams best-to-worst using FIFA tiebreakers.

    Primary: points, goal difference, goals scored. Ties are then broken on the
    head-to-head mini-table (points, GD, GF) among only the tied teams, and finally
    by a random draw (``rng``) standing in for fair-play/drawing of lots.
    """

    def primary_key(r: TeamRecord):
        return (r.points, r.gd, r.gf)

    ordered = sorted(records, key=primary_key, reverse=True)
    resolved: list[TeamRecord] = []
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and primary_key(ordered[j]) == primary_key(ordered[i]):
            j += 1
        cluster = ordered[i:j]
        resolved.extend(cluster if len(cluster) == 1 else _break_tie(cluster, rng))
        i = j
    return resolved


def _break_tie(cluster: list[TeamRecord], rng) -> list[TeamRecord]:
    """Break a tie among `cluster` using only their head-to-head matches, then randomly."""
    names = {r.team for r in cluster}

    def h2h_key(r: TeamRecord):
        pts = gf = ga = 0
        for opp, (p, f, a) in r.h2h.items():
            if opp in names:
                pts += p
                gf += f
                ga += a
        return (pts, gf - ga, gf)

    ranked = sorted(cluster, key=h2h_key, reverse=True)
    # Final random shuffle within still-identical head-to-head keys.
    out: list[TeamRecord] = []
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and h2h_key(ranked[j]) == h2h_key(ranked[i]):
            j += 1
        tied = ranked[i:j]
        if len(tied) > 1:
            order = list(rng.permutation(len(tied)))
            tied = [tied[k] for k in order]
        out.extend(tied)
        i = j
    return out


def select_best_thirds(thirds: list[tuple[str, TeamRecord]], rng, n: int = 8) -> list[str]:
    """Pick the `n` best third-placed teams across all groups (points, GD, GF, random)."""

    def key(item):
        _, r = item
        return (r.points, r.gd, r.gf, rng.random())

    ranked = sorted(thirds, key=key, reverse=True)
    return [team for team, _ in ranked[:n]]


def _seeding_order(n: int) -> list[int]:
    """Standard single-elimination seeding positions for a bracket of size `n`
    (a power of two). Seed 1 and seed 2 can only meet in the final."""
    order = [0]
    while len(order) < n:
        size = len(order) * 2
        order = [v for pair in ((s, size - 1 - s) for s in order) for v in pair]
    return order


def build_bracket_order(qualifiers: list[tuple[str, tuple]]) -> list[str]:
    """Return 32 team names in bracket-position order, strongest seeds spread apart.

    ``qualifiers`` is a list of ``(team, seed_key)`` where a larger ``seed_key`` is a
    stronger seed (e.g. ``(rank_class, points, gd, gf)`` with rank_class 2=winner).
    """
    ranked = [team for team, _ in sorted(qualifiers, key=lambda x: x[1], reverse=True)]
    positions = _seeding_order(len(ranked))
    slot = [""] * len(ranked)
    for seed_idx, team in enumerate(ranked):
        slot[positions.index(seed_idx)] = team
    return slot
