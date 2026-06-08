"""Tournament-stage reliability (Lane 3 #6) — do deep-run probabilities mean what they say?

Match-level calibration is validated (ECE ~0.03) and ``calibration.py`` checks the *qualification*
stage. This goes the whole way up the bracket: for each reconstructable 32-team World Cup it
simulates the full tournament (group stage → the canonical 8-group knockout bracket) to get every
team's P(reach the quarter-final / semi-final / final / title), then checks those probabilities
against who *actually* got there. Pooling several editions across all stages gives a reliability
curve for exactly the kind of number the headline forecast sells — "Argentina 22% to win" — which
match RPS never tests.

The bracket pairing uses the date-ordered group labels A–H (a valid WC structure), so this is an
*aggregate* reliability check over many team-stage cases, not a team-exact path reconstruction.
Honest caveat stands: a handful of tournaments is thin (champion-level calibration, gap #35, is
inherent), but more stages × more editions is materially more evidence than two World Cups.

Run via ``stage_reliability.py``.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from ..data import loaders
from ..model.match_model import MatchModel
from ..ratings.dixon_coles import DixonColesModel
from ..ratings.elo import EloModel
from ..simulate import format as fmt
from .calibration import reconstruct_groups

# 32-team World Cups with the stable 8-group format and enough prior history to rate every team.
WC_EDITIONS: dict[int, date] = {
    2010: date(2010, 6, 11),
    2014: date(2014, 6, 12),
    2018: date(2018, 6, 14),
    2022: date(2022, 11, 20),
}

# Reach-stage labels (each = "reached at least this round"). "qualify" = made the Round of 16.
STAGES = ["qualify", "qf", "sf", "final", "champion"]


def _shootout_winners(path=None) -> dict[tuple, str]:
    import pandas as pd

    sp = (Path(path).parent if path else loaders.DEFAULT_RESULTS.parent) / "shootouts.csv"
    out: dict[tuple, str] = {}
    if sp.exists():
        for r in pd.read_csv(sp).itertuples(index=False):
            out[(str(r.date), r.home_team, r.away_team)] = r.winner
    return out


def actual_stages(year: int, path=None) -> dict[str, set[str]] | None:
    """Who actually reached each stage, parsed from results.csv (score + shootout resolution)."""
    import pandas as pd

    df = pd.read_csv(loaders.DEFAULT_RESULTS if path is None else path)
    df = df[df["tournament"] == "FIFA World Cup"].copy()
    df["date"] = pd.to_datetime(df["date"])
    wc = df[df["date"].dt.year == year].sort_values("date")
    if len(wc) < 64:
        return None
    rows = list(wc.itertuples(index=False))
    ko_rows = rows[48:64]  # skip the 8×6 group games; the 16 knockout games follow
    shootouts = _shootout_winners(path)

    def winner(r) -> str:
        hs, as_ = int(r.home_score), int(r.away_score)
        if hs > as_:
            return r.home_team
        if as_ > hs:
            return r.away_team
        return (shootouts.get((str(r.date.date()), r.home_team, r.away_team))
                or shootouts.get((str(r.date.date()), r.away_team, r.home_team)))

    qualify = {t for r in ko_rows for t in (r.home_team, r.away_team)}
    r16, qf, sf = ko_rows[:8], ko_rows[8:12], ko_rows[12:14]
    final_row = ko_rows[-1]  # the final is the last knockout match
    qf_teams = {winner(r) for r in r16}    # won the Round of 16 → quarter-finalists
    sf_teams = {winner(r) for r in qf}     # won the quarter-final → semi-finalists
    finalists = {final_row.home_team, final_row.away_team}
    champ = winner(final_row)
    # Sanity: the two finalists are exactly the semi-final winners, and the champion is a finalist.
    if len(qualify) != 16 or champ not in finalists or not finalists <= {winner(r) for r in sf}:
        return None
    return {"qualify": qualify, "qf": qf_teams, "sf": sf_teams,
            "final": finalists, "champion": {champ}}


def simulate_stages(
    model: MatchModel, groups: dict[str, list[str]], n_sims: int = 20000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """P(reach each stage) for every team via full group+bracket Monte Carlo."""
    rng = np.random.default_rng(seed)
    teams = [t for ts in groups.values() for t in ts]
    counts = {s: dict.fromkeys(teams, 0) for s in STAGES}

    def ko(x, y):
        return model.sample_knockout(x, y, rng, neutral=True)

    for _ in range(n_sims):
        win, run = {}, {}
        for g, ts in groups.items():
            recs = {t: fmt.TeamRecord(t) for t in ts}
            for ia, ib in fmt.ROUND_ROBIN_PAIRS:
                ga, gb = model.sample_score(ts[ia], ts[ib], rng, neutral=True)
                recs[ts[ia]].add_match(ts[ib], ga, gb)
                recs[ts[ib]].add_match(ts[ia], gb, ga)
            ranked = fmt.rank_group(list(recs.values()), rng)
            win[g], run[g] = ranked[0].team, ranked[1].team
            counts["qualify"][ranked[0].team] += 1
            counts["qualify"][ranked[1].team] += 1
        # Canonical 8-group bracket (A..H by date order): winners cross with the next group's runner.
        r16 = [(win["A"], run["B"]), (win["C"], run["D"]), (win["E"], run["F"]), (win["G"], run["H"]),
               (win["B"], run["A"]), (win["D"], run["C"]), (win["F"], run["E"]), (win["H"], run["G"])]
        qf_w = [ko(x, y) for x, y in r16]
        for t in qf_w:
            counts["qf"][t] += 1
        sf_w = [ko(qf_w[i], qf_w[i + 1]) for i in range(0, 8, 2)]
        for t in sf_w:
            counts["sf"][t] += 1
        fin = [ko(sf_w[0], sf_w[1]), ko(sf_w[2], sf_w[3])]
        for t in fin:
            counts["final"][t] += 1
        counts["champion"][ko(fin[0], fin[1])] += 1

    return {s: {t: counts[s][t] / n_sims for t in teams} for s in STAGES}


def backtest(editions: dict[int, date] | None = None, n_sims: int = 20000) -> dict:
    """Pool (predicted, actual) per stage over the editions whose groups+bracket reconstruct cleanly."""
    editions = editions or WC_EDITIONS
    allm = loaders.load_results(since="2002-01-01", min_team_matches=15)
    per_stage = {s: {"pred": [], "actual": []} for s in STAGES}
    used = []
    for yr, start in editions.items():
        groups = reconstruct_groups(yr)
        actual = actual_stages(yr)
        if len(groups) != 8 or actual is None:
            continue
        train = [m for m in allm if m["date"] < start]
        elo = EloModel().fit(train)
        mm = float(np.mean(list(elo.ratings.values())))
        dc = DixonColesModel(half_life_days=1100.0)
        dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
        teams = [t for ts in groups.values() for t in ts]
        if any(t not in dc.params.attack for t in teams):
            continue
        pred = simulate_stages(MatchModel(dc), groups, n_sims=n_sims)
        for s in STAGES:
            for t in teams:
                per_stage[s]["pred"].append(pred[s][t])
                per_stage[s]["actual"].append(1.0 if t in actual[s] else 0.0)
        used.append(yr)
    return {"editions": used, "per_stage": per_stage}


def _binary_ece(pred: np.ndarray, actual: np.ndarray, n_bins: int = 5) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(pred[m].mean() - actual[m].mean())
    return float(total / len(pred))


def summarize(bt: dict) -> dict:
    """Per-stage Brier, ECE, base rate, mean predicted prob, and a pooled summary."""
    out = {"editions": bt["editions"], "stages": {}}
    all_pred, all_act = [], []
    for s in STAGES:
        pred = np.array(bt["per_stage"][s]["pred"])
        act = np.array(bt["per_stage"][s]["actual"])
        all_pred.append(pred)
        all_act.append(act)
        out["stages"][s] = {
            "n": int(len(pred)),
            "base_rate": float(act.mean()),
            "mean_pred": float(pred.mean()),
            "brier": float(np.mean((pred - act) ** 2)),
            "ece": _binary_ece(pred, act),
        }
    p, a = np.concatenate(all_pred), np.concatenate(all_act)
    out["pooled"] = {"n": int(len(p)), "brier": float(np.mean((p - a) ** 2)), "ece": _binary_ece(p, a)}
    return out
