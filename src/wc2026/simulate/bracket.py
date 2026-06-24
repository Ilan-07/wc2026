"""Official WC2026 Round-of-32 bracket template (parsed from the published knockout structure).

This removes the "wait for the bracket" problem: the slot layout is fixed and known *now* (only the
*teams* filling the slots depend on group results, which is unavoidable). Each of the 32 slots, in
bracket order, is one of:
  * ('W', group)              — that group's winner
  * ('R', group)              — that group's runner-up
  * ('3', {candidate groups}) — a best-third from one of those groups (FIFA allocation)

``build_official_bracket`` fills the template from group standings, resolving the third-place slots
via FIFA's official Annexe-C allocation table. The result is the real bracket order, auto-assembled.

Source: FWC26 Regulations, Article 12.6 (Round-of-32 fixture list) and Annexe C (third allocation).
"""

from __future__ import annotations

from .third_allocation import allocate_thirds

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

# Each third-slot sits opposite a group winner in its Round-of-32 match; that winner's column in
# Annexe C tells us which third belongs in the slot. winner-group letter -> third-slot index.
_SLOT_BY_WINNER: dict[str, int] = {
    SLOT_SPECS[i - 1 if i % 2 else i + 1][1]: i for i in THIRD_SLOTS
}


def _match_thirds(qualified_third_groups: list[str]) -> dict[int, str]:
    """Map each qualifying third's group to its third-slot using FIFA's Annexe-C table.

    The candidate sets only constrain the assignment; FIFA's published table fixes it exactly.
    Returns ``slot_index -> group``.
    """
    winner_to_third = allocate_thirds(qualified_third_groups)  # winner group -> third group
    return {_SLOT_BY_WINNER[w]: g for w, g in winner_to_third.items()}


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
