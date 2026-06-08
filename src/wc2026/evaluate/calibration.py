"""Tournament-stage calibration backtest (gap #1) — does the simulator's *tournament* output
mean what it says, not just its match output?

Match-level calibration is already validated (ECE ~0.03). This goes a level up: for the 2018 and
2022 World Cups, simulate the group stage many times to get each team's P(reach the knockout =
finish top-2), and check those probabilities against who *actually* qualified. Pooling 2 WCs × 32
teams gives a real (if small) reliability curve for a *tournament-stage* probability — the kind of
number the headline forecast produces and that match-level RPS doesn't test.

Honest caveat: only ~2 tournaments of qualification outcomes exist, so this narrows the
champion-calibration blind spot but cannot fully close it (gap #35 is inherent).
"""

from __future__ import annotations

from datetime import date

import numpy as np

from ..data import loaders
from ..model.match_model import MatchModel
from ..ratings.dixon_coles import DixonColesModel
from ..ratings.elo import EloModel
from ..simulate import format as fmt

WCS = {2018: date(2018, 6, 14), 2022: date(2022, 11, 20)}


def reconstruct_groups(year: int, n_group_matches: int = 48, path=None) -> dict[str, list[str]]:
    """Reconstruct the 8 groups of a 32-team World Cup from its first 48 (group-stage) fixtures."""
    import pandas as pd

    df = pd.read_csv(loaders.DEFAULT_RESULTS if path is None else path)
    df = df[(df["tournament"] == "FIFA World Cup")].copy()
    df["date"] = pd.to_datetime(df["date"])
    wc = df[df["date"].dt.year == year].sort_values("date").head(n_group_matches)
    adj: dict[str, set] = {}
    first: dict[str, object] = {}
    for r in wc.itertuples(index=False):
        a, b = r.home_team, r.away_team
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
        for t in (a, b):
            if t not in first or r.date < first[t]:
                first[t] = r.date
    seen, comps = set(), []
    for t in adj:
        if t in seen:
            continue
        stack, comp = [t], []
        while stack:
            c = stack.pop()
            if c in seen:
                continue
            seen.add(c); comp.append(c)
            stack.extend(adj[c] - seen)
        if len(comp) == 4:
            comps.append(comp)
    comps.sort(key=lambda c: min(first[t] for t in c))
    return {chr(ord("A") + i): sorted(c) for i, c in enumerate(comps)}


def _actual_top2(year: int, groups: dict[str, list[str]], path=None) -> set[str]:
    """Teams that actually finished top-2 of their group (reached the knockout)."""
    import pandas as pd

    df = pd.read_csv(loaders.DEFAULT_RESULTS if path is None else path)
    df = df[df["tournament"] == "FIFA World Cup"].copy()
    df["date"] = pd.to_datetime(df["date"])
    gof = {t: g for g, ts in groups.items() for t in ts}
    recs = {t: fmt.TeamRecord(t) for ts in groups.values() for t in ts}
    for r in df[(df["date"].dt.year == year)].sort_values("date").head(48).itertuples(index=False):
        if gof.get(r.home_team) == gof.get(r.away_team) and gof.get(r.home_team) is not None:
            recs[r.home_team].add_match(r.away_team, int(r.home_score), int(r.away_score))
            recs[r.away_team].add_match(r.home_team, int(r.away_score), int(r.home_score))
    rng = np.random.default_rng(0)
    top2 = set()
    for ts in groups.values():
        ranked = fmt.rank_group([recs[t] for t in ts], rng)
        top2.update(r.team for r in ranked[:2])
    return top2


def simulate_qualification(model: MatchModel, groups: dict[str, list[str]],
                           n_sims: int = 20000, seed: int = 0) -> dict[str, float]:
    """P(each team finishes top-2 of its group), from many group-stage simulations."""
    rng = np.random.default_rng(seed)
    teams = [t for ts in groups.values() for t in ts]
    count = {t: 0 for t in teams}
    for _ in range(n_sims):
        for ts in groups.values():
            recs = {t: fmt.TeamRecord(t) for t in ts}
            for ia, ib in fmt.ROUND_ROBIN_PAIRS:
                ga, gb = model.sample_score(ts[ia], ts[ib], rng, neutral=True)
                recs[ts[ia]].add_match(ts[ib], ga, gb)
                recs[ts[ib]].add_match(ts[ia], gb, ga)
            for r in fmt.rank_group(list(recs.values()), rng)[:2]:
                count[r.team] += 1
    return {t: c / n_sims for t, c in count.items()}


def backtest(n_sims: int = 20000) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted P(qualify), actual 0/1) pooled over the 2018 & 2022 World Cups."""
    allm = loaders.load_results(since="2006-01-01", min_team_matches=15)
    preds, actuals = [], []
    for yr, start in WCS.items():
        groups = reconstruct_groups(yr)
        if len(groups) != 8:
            continue
        train = [m for m in allm if m["date"] < start]
        elo = EloModel().fit(train); mm = float(np.mean(list(elo.ratings.values())))
        dc = DixonColesModel(half_life_days=1100.0)
        dc.fit(train, init_attack={t: (r - mm) / 400.0 for t, r in elo.ratings.items()})
        teams = [t for ts in groups.values() for t in ts]
        if any(t not in dc.params.attack for t in teams):
            continue
        p = simulate_qualification(MatchModel(dc), groups, n_sims=n_sims)
        top2 = _actual_top2(yr, groups)
        for t in teams:
            preds.append(p[t]); actuals.append(1.0 if t in top2 else 0.0)
    return np.array(preds), np.array(actuals)


def reliability_plot(preds, actuals, out="data/processed/calibration.png", n_bins: int = 5):
    """Render a reliability curve (predicted vs observed qualification rate) to PNG."""
    import matplotlib
    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(preds, bins) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            xs.append(preds[m].mean()); ys.append(actuals[m].mean()); ns.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", color="#2563eb", label="simulator")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), fontsize=8, xytext=(4, -10), textcoords="offset points")
    ax.set(xlabel="predicted P(qualify)", ylabel="observed qualify rate",
           title="WC2018+2022 group-qualification calibration", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return out
