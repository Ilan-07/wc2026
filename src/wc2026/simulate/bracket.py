"""Official WC2026 Round-of-32 bracket template (parsed from the published knockout structure).

This removes the "wait for the bracket" problem: the slot layout is fixed and known *now* (only the
*teams* filling the slots depend on group results, which is unavoidable). Each of the 32 slots, in
bracket order, is one of:
  * ('W', group)              — that group's winner
  * ('R', group)              — that group's runner-up
  * ('3', {candidate groups}) — a best-third from one of those groups (FIFA allocation)

``build_official_bracket`` fills the template from group standings, resolving the third-place slots
by constraint matching. The result is the real bracket order, auto-assembled — no manual file.

Source: en.wikipedia.org "2026 FIFA World Cup knockout stage" (Round of 32 fixture list).
"""

from __future__ import annotations

# 32 slots in bracket order; adjacent pairs (0,1),(2,3),... are the Round-of-32 matches.
SLOT_SPECS: list[tuple] = [
    ("W", "E"), ("3", frozenset("ABCDF")),
    ("W", "I"), ("3", frozenset("CDFGH")),
    ("R", "A"), ("R", "B"),
    ("W", "F"), ("R", "C"),
    ("R", "K"), ("R", "L"),
    ("W", "H"), ("R", "J"),
    ("W", "D"), ("3", frozenset("BEFIJ")),
    ("W", "G"), ("3", frozenset("AEHIJ")),
    ("W", "C"), ("R", "F"),
    ("R", "E"), ("R", "I"),
    ("W", "A"), ("3", frozenset("CEFHI")),
    ("W", "L"), ("3", frozenset("EHIJK")),
    ("W", "J"), ("R", "H"),
    ("R", "D"), ("R", "G"),
    ("W", "B"), ("3", frozenset("EFGIJ")),
    ("W", "K"), ("3", frozenset("DEIJL")),
]

THIRD_SLOTS = [i for i, s in enumerate(SLOT_SPECS) if s[0] == "3"]


def _match_thirds(qualified_third_groups: list[str]) -> dict[int, str]:
    """Assign each qualified third's group to a third-slot honoring candidate constraints.

    Bipartite (augmenting-path) matching of the 8 qualifying groups to the 8 third-slots; FIFA's
    candidate sets are designed so a valid perfect matching always exists.
    """
    cands = {i: SLOT_SPECS[i][1] for i in THIRD_SLOTS}
    slot_of_group: dict[str, int] = {}

    def aug(group, visited):
        for s in THIRD_SLOTS:
            if group in cands[s] and s not in visited:
                visited.add(s)
                cur = next((g for g, ss in slot_of_group.items() if ss == s), None)
                if cur is None or aug(cur, visited):
                    slot_of_group[group] = s
                    return True
        return False

    for g in qualified_third_groups:
        aug(g, set())
    return {s: g for g, s in slot_of_group.items()}


def build_official_bracket(
    winners: dict[str, str], runners: dict[str, str], thirds_by_group: dict[str, str]
) -> list[str]:
    """Fill the official template -> 32 team names in bracket order.

    ``winners``/``runners``: group -> team. ``thirds_by_group``: the 8 qualifying thirds, group -> team.
    """
    slot_group = _match_thirds(list(thirds_by_group))  # third-slot index -> group
    bracket: list[str] = []
    for i, (kind, val) in enumerate(SLOT_SPECS):
        if kind == "W":
            bracket.append(winners[val])
        elif kind == "R":
            bracket.append(runners[val])
        else:  # third slot
            bracket.append(thirds_by_group[slot_group[i]])
    return bracket
