"""Live tournament track record — grade the model against WC2026 results as they're played.

Leakage-free by construction: it grades the **production** match model (the same rating the
dashboard ships — Bayesian Poisson + xG blend + goals recalibration, via ``predict.build_match_model``
so it can't drift from the forecast), **frozen on data before kickoff**, with host advantage and
altitude applied per fixture exactly like the live simulator. Each WC2026 match is then scored as it
happens (proper W/D/L Ranked Probability Score vs a uniform baseline). Populates from June 11; before
then it honestly reports that scoring hasn't started.

Hit-rate is reported as **decisive-match accuracy** (draws excluded — a draw is almost never the
single most-likely outcome, so it is structurally unpickable; see ``wc2026.evaluate.pick``).

    PYTHONPATH=src python cli.py track     # update + print the live record

Writes ``data/processed/track_record.json`` (published alongside the dashboard) and is embedded in
the dashboard payload (`data["track"]`) so the page shows a live, honest receipt of how the forecast
is actually doing — the credibility multiplier.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from wc2026.data import loaders
from wc2026.evaluate.metrics import summary as score_summary
from wc2026.evaluate.pick import decisive_accuracy, pick
from wc2026.reports.scores import _group_extra, score_fixture

TRACK_FILE = Path("data/processed/track_record.json")
KICKOFF = dt.date(2026, 6, 11)
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def _outcome(home_goals: int, away_goals: int) -> int:
    """W/D/L outcome index matching MatchModel.wdl order: 0 home win, 1 draw, 2 away win."""
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def _build_model(train: list[dict], xg_before: str, bayesian: bool):
    """The production central match model, frozen on pre-kickoff history (leakage-free).

    Thin seam over ``predict.build_match_model`` (imported lazily to avoid a heavy import on the
    cheap pre-kickoff path) so tests can stub the model without fitting. ``xg_before`` keeps the
    xG blend out-of-sample too.
    """
    from predict import build_match_model
    return build_match_model(train, bayesian=bayesian, xg_before=xg_before)


def _as_date(x):
    """Normalise pandas Timestamp / datetime to a plain date (loaders mix the two)."""
    m = getattr(x, "date", None)
    return m() if callable(m) else x


def track_record(write: bool = True, bayesian: bool = True) -> dict:
    """Score the pre-kickoff-frozen model on every WC2026 match played so far.

    Grades the **production** match model (the rating the dashboard ships: Bayesian Poisson + xG
    blend + goals recalibration, with host advantage and altitude applied per fixture), frozen on
    history before kickoff so there is no look-ahead. ``bayesian=False`` forces the MLE fallback.
    """
    groups = loaders.load_wc2026_groups()
    teams = {t for g in groups.values() for t in g}
    allm = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=teams)
    start = _as_date(min((d for d, *_ in loaders.load_wc2026_group_fixtures()), default=KICKOFF))

    played = [m for m in allm
              if _as_date(m["date"]) >= start and m["importance"] == "world_cup"
              and m["home_team"] in teams and m["away_team"] in teams]

    rec: dict = {
        "kickoff": start.isoformat(),
        "updated": dt.datetime.now().isoformat(timespec="seconds"),
        "n_matches": len(played),
    }
    if not played:
        rec.update({"status": "pending", "note": f"Live scoring begins at kickoff ({start.isoformat()})."})
        if write:
            _write(rec)
        return rec

    # Freeze the production model on pre-kickoff history only — no look-ahead into tournament results.
    train = [m for m in allm if _as_date(m["date"]) < start]
    model = _build_model(train, xg_before=start.isoformat(), bayesian=bayesian)
    venue_alt = loaders.load_wc2026_group_venue_altitudes()  # altitude shift at the Mexican venues

    probs, outs, correct = [], [], 0
    for m in played:
        # score_fixture applies host advantage (host treated as home) + altitude exactly like the
        # live simulator/dashboard, and returns W/D/L oriented to the listed home/away teams.
        extra = _group_extra(m["home_team"], m["away_team"], venue_alt, None)
        fs = score_fixture(model, m["home_team"], m["away_team"], extra_home=extra)
        p = [fs.p_home, fs.p_draw, fs.p_away]
        o = _outcome(m["home_score"], m["away_score"])
        probs.append(p)
        outs.append(o)
        correct += int(pick(p) == o)  # draw-aware pick (defaults to argmax in practice)

    probs_a, outs_a = np.array(probs), np.array(outs)
    s = score_summary(probs_a, outs_a)
    uni = score_summary(np.array([UNIFORM] * len(outs)), outs_a)
    # The honest hit-rate: a draw is almost never the single most-likely outcome, so the pick can
    # only ever be "right" on matches that had a winner. Report accuracy on those decisive matches
    # alongside the raw count so a glut of (unpickable) draws can't masquerade as a model failure.
    dec_correct, n_decisive = decisive_accuracy(probs_a, outs_a)
    rec.update({
        "status": "live",
        "rps": round(s["rps"], 4),
        "uniform_rps": round(uni["rps"], 4),
        "skill": round(uni["rps"] - s["rps"], 4),
        "log_loss": round(s["log_loss"], 4),
        "brier": round(s["brier"], 4),
        "calls_correct": correct,
        "calls_decisive": dec_correct,
        "n_decisive": n_decisive,
        "n_draws": int((outs_a == 1).sum()),
    })
    if write:
        _write(rec)
    return rec


def _write(rec: dict) -> None:
    TRACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACK_FILE.write_text(json.dumps(rec, indent=2))


def main() -> None:
    rec = track_record(write=True)
    if rec["n_matches"] == 0:
        print(f"Live track record: {rec['note']}")
    else:
        print(f"Live track record over {rec['n_matches']} WC2026 matches:")
        print(f"  RPS {rec['rps']} vs uniform {rec['uniform_rps']}  (skill {rec['skill']:+})")
        print(f"  decisive calls {rec['calls_decisive']}/{rec['n_decisive']}  "
              f"({rec['n_draws']} draws excluded — unpickable)  |  "
              f"raw calls {rec['calls_correct']}/{rec['n_matches']}")
        print(f"  log-loss {rec['log_loss']}  brier {rec['brier']}")


if __name__ == "__main__":
    main()
