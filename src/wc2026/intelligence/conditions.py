"""Playing-conditions engine: altitude & travel (plan Phase B / Tier 2).

Altitude is the one environmental factor with real physiological evidence: sea-level teams tire
faster in thin air, while teams that train at altitude are acclimatised. WC2026 has three Mexican
venues — Estadio Azteca (Mexico City, 2,240 m) most of all. Travel/fatigue is a second, weaker
factor across a continent-sized tournament.

Like injuries, these are **principled prior adjustments**, not RPS-gate-passed features — there are
too few high-altitude WC matches to validate the magnitude. They plug into the MatchModel adjustment
hook for a specific venue, and are exposed as a transparent, tunable signal.
"""

from __future__ import annotations

import math

from ..data.venues import Venue, home_altitude


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def altitude_penalty(team: str, venue: Venue, per_1000m: float = 0.12) -> float:
    """Log-rate penalty for ``team`` playing at ``venue`` above its acclimatised altitude.

    penalty = per_1000m · max(0, venue_alt − home_alt) / 1000. A sea-level side at Azteca
    (2,240 m) gets ≈ 0.27 (a real but not decisive handicap); an Andean side gets ≈ 0.
    """
    excess = max(0.0, venue.altitude_m - home_altitude(team))
    return per_1000m * excess / 1000.0


class AltitudeAdjustment:
    """MatchModel adjustment for a fixture at a specific venue.

    delta = penalty_away − penalty_home, so the side less suited to the altitude scores less.
    """

    def __init__(self, venue: Venue, per_1000m: float = 0.12):
        self.venue = venue
        self.per_1000m = per_1000m

    def __call__(self, home: str, away: str) -> float:
        ph = altitude_penalty(home, self.venue, self.per_1000m)
        pa = altitude_penalty(away, self.venue, self.per_1000m)
        return pa - ph


def altitude_table(teams: list[str], venue: Venue, per_1000m: float = 0.12) -> list[tuple[str, float]]:
    """Teams ranked by altitude handicap at a venue (most disadvantaged first)."""
    return sorted(((t, altitude_penalty(t, venue, per_1000m)) for t in teams),
                  key=lambda x: x[1], reverse=True)


# --------------------------------------------------------------------------- fatigue
# Rest-days congestion and cumulative travel are the one factor that is *genuinely orthogonal*
# to a results-based rating: a strong team forced into a 3-day turnaround after a cross-continent
# flight is not weaker on paper, only on the night. Unlike squad reputation / FIFA rank (mirrors of
# strength the rating already has), this is information NOT in past results — so it is worth testing.
# It is a tunable prior, ablation-gated like everything else (see fatigue_ablation.py); the rest-days
# component is gateable on 2018/2022 (dates only), travel needs venue coords we have only for WC2026.

BASELINE_REST_DAYS = 4.0  # a normal group-stage turnaround; shorter than this accrues a deficit


def fatigue_penalty(
    rest_days: float,
    cum_travel_km: float = 0.0,
    *,
    rest_k: float = 0.04,
    travel_per_1000km: float = 0.02,
    baseline_rest: float = BASELINE_REST_DAYS,
) -> float:
    """Log-rate fatigue penalty: short rest + accumulated travel make a side score slightly less.

    penalty = rest_k · max(0, baseline_rest − rest_days) + travel_per_1000km · cum_travel_km/1000.
    A team on a 2-day turnaround (deficit 2) after 6,000 km of travel gets ≈ 0.08 + 0.12 = 0.20 —
    a modest handicap, comparable in scale to the altitude effect. Both ``_k`` magnitudes are priors.
    """
    rest_deficit = max(0.0, baseline_rest - float(rest_days))
    return rest_k * rest_deficit + travel_per_1000km * (max(0.0, cum_travel_km) / 1000.0)


class FatigueAdjustment:
    """MatchModel adjustment for a fixture, given each side's precomputed fatigue penalty.

    delta = penalty_away − penalty_home, so the more fatigued side scores less (antisymmetric,
    exactly like :class:`AltitudeAdjustment`). ``pens`` maps team -> penalty for *this* fixture.
    """

    def __init__(self, pens: dict[str, float]):
        self.pens = pens

    def __call__(self, home: str, away: str) -> float:
        return self.pens.get(away, 0.0) - self.pens.get(home, 0.0)


def fixture_fatigue_penalties(
    fixtures: list[tuple],
    *,
    rest_k: float = 0.04,
    travel_per_1000km: float = 0.02,
    use_travel: bool = True,
) -> dict[frozenset, dict[str, float]]:
    """Map each fixture ``frozenset({home, away})`` -> ``{team: fatigue_penalty}``.

    ``fixtures`` is a chronological list of ``(date, home, away, city)`` rows. For every team
    appearance, rest-days = gap since that team's previous fixture (the first match has no deficit),
    and cumulative travel = great-circle km over the team's venue sequence up to and including this
    fixture. Travel is added only when cities resolve to known venues (``use_travel``); an unknown
    city simply skips the travel term, so the rest-days term still applies (this is what lets the
    2018/2022 ablation gate the rest-days component without historical venue coordinates).
    """
    from ..data.venues import venue_for_city

    rows = sorted(fixtures, key=lambda r: r[0])
    # Per-team running state while walking the schedule in date order.
    prev_date: dict[str, object] = {}
    prev_v: dict[str, Venue] = {}
    cum_km: dict[str, float] = {}

    def step(team: str, date, city) -> float:
        rest = BASELINE_REST_DAYS if team not in prev_date else _days_between(prev_date[team], date)
        v = venue_for_city(str(city)) if use_travel else None
        if v is not None and prev_v.get(team) is not None:
            pv = prev_v[team]
            cum_km[team] = cum_km.get(team, 0.0) + haversine_km(pv.lat, pv.lon, v.lat, v.lon)
        prev_date[team] = date
        if v is not None:
            prev_v[team] = v
        return fatigue_penalty(rest, cum_km.get(team, 0.0),
                               rest_k=rest_k, travel_per_1000km=travel_per_1000km)

    out: dict[frozenset, dict[str, float]] = {}
    for date, home, away, city in rows:
        ph = step(home, date, city)
        pa = step(away, date, city)
        out[frozenset((home, away))] = {home: ph, away: pa}
    return out


def _days_between(d0, d1) -> float:
    """Days between two date-like values (pandas Timestamp / datetime / date)."""
    try:
        return abs(float((d1 - d0).days))
    except AttributeError:  # plain difference already a number
        return abs(float(d1 - d0))
