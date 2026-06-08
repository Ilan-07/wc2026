"""Explanation & reasoning layer — the cathedral's *valid* purpose: traceability (plan Tier F).

Per the validity framing, these layers do NOT improve the forecast. Their job is to make every
prediction **auditable**: decompose a team's number into the rating, the path, the market
disagreement, the squad, and the conditions — each pointing at the data behind it.

The "agents" are a **deterministic reasoning pipeline**, not LLM agents: Analyst, Market-watcher,
Contrarian and Judge are plain functions that each emit a structured, evidence-backed claim, and
the Judge reconciles them. (An LLM could later phrase these; the logic stays deterministic and
inspectable.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.venues import home_altitude
from ..graph.kg import KnowledgeGraph


@dataclass
class ReportContext:
    teams: list[str]
    elo: dict[str, float]            # team -> Elo
    groups: dict[str, list[str]]
    model_p: dict[str, float]        # champion probability (model)
    market_p: dict[str, float]       # champion probability (market)
    blended_p: dict[str, float]
    kg: KnowledgeGraph
    caps: dict[str, dict[str, int]]  # team -> {player: caps}
    sd: dict[str, float] = field(default_factory=dict)
    altitude_cities: dict[str, set] = field(default_factory=dict)  # team -> set of venue cities at altitude


@dataclass
class Claim:
    role: str
    text: str
    evidence: str


def _rank(d: dict[str, float], team: str) -> int:
    return sorted(d, key=lambda t: d[t], reverse=True).index(team) + 1


def _path_difficulty(ctx: ReportContext, team: str) -> tuple[str, float, list[str]]:
    g = ctx.kg.group_of(team) or "?"
    opps = sorted(o for o in ctx.groups.get(g, []) if o != team)
    avg = sum(ctx.elo.get(o, 1500) for o in opps) / max(len(opps), 1)
    return g, avg, opps


def analyst(ctx: ReportContext, team: str) -> Claim:
    er = _rank(ctx.elo, team)
    g, diff, opps = _path_difficulty(ctx, team)
    return Claim(
        "Analyst",
        f"Rated #{er}/48 by Elo ({ctx.elo.get(team,0):.0f}). Group {g} vs {', '.join(opps)}; "
        f"average opponent Elo {diff:.0f} ({'tough' if diff>1700 else 'moderate' if diff>1550 else 'soft'} draw).",
        "ratings.elo + group draw",
    )


def market_watcher(ctx: ReportContext, team: str) -> Claim:
    m, k = ctx.model_p.get(team, 0), ctx.market_p.get(team, 0)
    d = k - m
    verdict = ("crowd rates higher (possible overhype)" if d > 0.01
               else "model rates higher (possible value)" if d < -0.01 else "model and market agree")
    return Claim(
        "Market", f"Model {m:.1%} vs market {k:.1%} → {verdict} (Δ {d*100:+.1f}pp).",
        "fusion.divergence + live odds",
    )


def contrarian(ctx: ReportContext, team: str) -> Claim:
    m, k = ctx.model_p.get(team, 0), ctx.market_p.get(team, 0)
    if m > k + 0.01:
        txt = ("The model is more bullish than the market — its case rests on recent results/Elo, "
               "which can lag squad quality or breaking news the market prices in. Treat as a "
               "contrarian call, not consensus.")
    elif k > m + 0.01:
        txt = ("The market is more bullish — likely pricing squad reputation the results-based model "
               "discounts. The model's caution may be right if results genuinely lag the name value.")
    else:
        txt = "Little disagreement to challenge; model and market corroborate each other."
    return Claim("Contrarian", txt, "model vs de-vigged market")


def judge(ctx: ReportContext, team: str, claims: list[Claim]) -> Claim:
    b = ctx.blended_p.get(team, 0)
    sd = ctx.sd.get(team, 0)
    d = abs(ctx.market_p.get(team, 0) - ctx.model_p.get(team, 0))
    # confidence: lower when uncertainty band is wide relative to the estimate, or disagreement large
    conf = "high" if (sd < 0.25 * max(b, 1e-9) and d < 0.04) else "low" if (sd > 0.6 * max(b, 1e-9) or d > 0.08) else "medium"
    return Claim(
        "Judge",
        f"Reconciled title probability {b:.1%} (±{sd:.1%}). Confidence {conf} — "
        f"{'model and market align' if d<0.03 else 'model and market disagree, blend splits the difference'}.",
        "logarithmic opinion pool",
    )


@dataclass
class TeamReport:
    team: str
    claims: list[Claim]
    facts: dict[str, str]

    def render(self) -> str:
        lines = [f"=== {self.team} — intelligence report ==="]
        for k, v in self.facts.items():
            lines.append(f"  {k}: {v}")
        lines.append("  reasoning:")
        for c in self.claims:
            lines.append(f"    [{c.role}] {c.text}")
            lines.append(f"        ↳ {c.evidence}")
        return "\n".join(lines)


def explain_team(ctx: ReportContext, team: str) -> TeamReport:
    """Run the deterministic reasoning pipeline and assemble a traceable report for one team."""
    caps = ctx.caps.get(team, {})
    key_players = sorted(caps, key=lambda p: caps[p], reverse=True)[:3]
    mates = ctx.kg.clubmates(team)
    biggest = max(mates.items(), key=lambda kv: len(kv[1]), default=(None, []))
    alt = ctx.altitude_cities.get(team, set())

    facts = {
        "forecast": f"model {ctx.model_p.get(team,0):.1%} / market {ctx.market_p.get(team,0):.1%} "
                    f"/ blended {ctx.blended_p.get(team,0):.1%}",
        "coach": ctx.kg.coach(team) or "—",
        "key players": ", ".join(f"{p} ({caps[p]})" for p in key_players) or "—",
        "chemistry": (f"{len(biggest[1])} players from {biggest[0]}" if biggest[0] else "dispersed squad"),
        "altitude exposure": (f"plays at {', '.join(sorted(alt))} (high altitude)" if alt
                              else "no altitude venues" + (" (sea-level squad)" if home_altitude(team) < 500 else "")),
    }
    claims = []
    claims.append(analyst(ctx, team))
    claims.append(market_watcher(ctx, team))
    claims.append(contrarian(ctx, team))
    claims.append(judge(ctx, team, claims))
    return TeamReport(team=team, claims=claims, facts=facts)
