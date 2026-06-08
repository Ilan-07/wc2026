"""Football knowledge graph — built for *traceability*, not prediction (plan Tier F, valid use).

Per the validity framing: a knowledge graph does not improve the forecast (it re-encodes data the
rating already saw). Its legitimate value is **auditability and queryability** — every entity and
relationship behind a prediction is explicit and walkable. This is a dependency-free triple store
(subject, relation, object) assembled from data already loaded: teams, players, coaches, clubs,
groups, venues, fixtures.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class KnowledgeGraph:
    # relation -> subject -> set(objects), and the reverse index
    _fwd: dict[str, dict[str, set]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    _rev: dict[str, dict[str, set]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(set)))
    node_type: dict[str, str] = field(default_factory=dict)

    def add(self, subj: str, rel: str, obj: str, subj_type="", obj_type="") -> None:
        self._fwd[rel][subj].add(obj)
        self._rev[rel][obj].add(subj)
        if subj_type:
            self.node_type[subj] = subj_type
        if obj_type:
            self.node_type[obj] = obj_type

    def objects(self, subj: str, rel: str) -> set:
        return set(self._fwd.get(rel, {}).get(subj, set()))

    def subjects(self, obj: str, rel: str) -> set:
        return set(self._rev.get(rel, {}).get(obj, set()))

    # -------- convenience queries (the point of the graph) -------------------
    def squad(self, team: str) -> set:
        return self.subjects(team, "PLAYS_FOR")

    def coach(self, team: str) -> str | None:
        c = self.subjects(team, "COACHES")
        return next(iter(c), None)

    def group_of(self, team: str) -> str | None:
        g = self.objects(team, "IN_GROUP")
        return next(iter(g), None)

    def opponents(self, team: str) -> set:
        return self.objects(team, "PLAYS") | self.subjects(team, "PLAYS")

    def clubmates(self, team: str) -> dict[str, list[str]]:
        """Within a national squad, players grouped by their shared club (the chemistry view)."""
        by_club: dict[str, list[str]] = defaultdict(list)
        for p in self.squad(team):
            for club in self.objects(p, "PLAYS_AT"):
                by_club[club].append(p)
        return {c: sorted(ps) for c, ps in by_club.items() if len(ps) > 1}

    def stats(self) -> dict[str, int]:
        nt = defaultdict(int)
        for t in self.node_type.values():
            nt[t] += 1
        edges = sum(len(o) for r in self._fwd.values() for o in r.values())
        return {"nodes": len(self.node_type), "edges": edges, **nt}


def build_kg(groups, squads, fixtures_venues=None) -> KnowledgeGraph:
    """Assemble the KG from loaded data.

    ``groups``: {group: [teams]}; ``squads``: {team: Squad}; ``fixtures_venues`` optional
    iterable of (home, away, city) for PLAYS / PLAYED_AT edges.
    """
    kg = KnowledgeGraph()
    for g, teams in groups.items():
        for t in teams:
            kg.add(t, "IN_GROUP", g, subj_type="Team", obj_type="Group")
    for team, sq in squads.items():
        if sq.coach:
            kg.add(sq.coach, "COACHES", team, subj_type="Coach", obj_type="Team")
        for p in sq.players:
            kg.add(p.name, "PLAYS_FOR", team, subj_type="Player", obj_type="Team")
            if p.club:
                kg.add(p.name, "PLAYS_AT", p.club, obj_type="Club")
    for home, away, city in (fixtures_venues or []):
        kg.add(home, "PLAYS", away, subj_type="Team", obj_type="Team")
        if city:
            kg.add(home, "PLAYED_AT", city, obj_type="Venue")
            kg.add(away, "PLAYED_AT", city, obj_type="Venue")
    return kg
