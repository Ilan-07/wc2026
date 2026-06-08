"""Squad-derived contextual features (plan P4, layers: cohesion, chemistry, experience).

Turns the parsed WC2026 squads into per-team, *measured* features that later enter the
log-rate equation as covariates (each ablation-gated). Nothing here is hand-assigned: every
number is computed from real squad facts (caps, clubs, ages, national goals).

Feature glossary
----------------
experience_caps      : mean national-team caps  -> tournament experience / squad cohesion
experience_top       : mean caps of the most-capped XI -> spine experience
shared_club_chem     : fraction of in-squad player *pairs* sharing a club -> chemistry proxy
                       (players who play together weekly already understand each other)
league_quality       : share of players at clubs in a top-5 European league -> club level
attack_pedigree      : total national goals among forwards/mids -> attacking track record
mean_age / age_spread: squad age profile -> cohesion vs renewal
n_clubs              : distinct clubs represented (low = concentrated core)

These are returned standardized-ready (raw values); the model layer z-scores them before
fitting weights, so a feature that does not improve RPS simply gets shrunk toward zero.
"""

from __future__ import annotations

from collections import Counter

from ..data.squads import Squad

# Clubs in these leagues are treated as top-tier (club_nat codes used by Wikipedia).
TOP5_LEAGUES = {"ENG", "ESP", "ITA", "GER", "FRA"}


def squad_features(squad: Squad) -> dict[str, float]:
    players = squad.players
    n = len(players)
    if n == 0:
        return {}

    caps = [p.caps for p in players]
    ages = [p.age for p in players if p.age is not None]

    # shared-club chemistry: among all C(n,2) pairs, how many share a club?
    club_counts = Counter(p.club for p in players if p.club)
    same_club_pairs = sum(c * (c - 1) // 2 for c in club_counts.values())
    total_pairs = n * (n - 1) // 2
    shared_club_chem = same_club_pairs / total_pairs if total_pairs else 0.0

    top_xi = sorted(caps, reverse=True)[:11]
    attackers = [p for p in players if p.pos in ("FW", "MF")]

    return {
        "experience_caps": sum(caps) / n,
        "experience_top": sum(top_xi) / len(top_xi) if top_xi else 0.0,
        "shared_club_chem": shared_club_chem,
        "league_quality": sum(1 for p in players if p.club_nat in TOP5_LEAGUES) / n,
        "attack_pedigree": float(sum(p.goals for p in attackers)),
        "mean_age": sum(ages) / len(ages) if ages else 0.0,
        "age_spread": (max(ages) - min(ages)) if ages else 0.0,
        "n_clubs": float(len(club_counts)),
    }


def all_squad_features(squads: dict[str, Squad]) -> dict[str, dict[str, float]]:
    return {team: squad_features(sq) for team, sq in squads.items()}
