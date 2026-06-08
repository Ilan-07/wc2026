"""Live tournament track record — grade the model against WC2026 results as they're played.

Leakage-free by construction: the rating is **frozen on data before kickoff**, then each WC2026
group match is scored as it happens (proper W/D/L Ranked Probability Score vs a uniform baseline).
Populates from June 11; before then it honestly reports that scoring hasn't started.

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
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel

TRACK_FILE = Path("data/processed/track_record.json")
KICKOFF = dt.date(2026, 6, 11)
UNIFORM = [1 / 3, 1 / 3, 1 / 3]


def _outcome(home_goals: int, away_goals: int) -> int:
    """W/D/L outcome index matching MatchModel.wdl order: 0 home win, 1 draw, 2 away win."""
    return 0 if home_goals > away_goals else (1 if home_goals == away_goals else 2)


def _as_date(x):
    """Normalise pandas Timestamp / datetime to a plain date (loaders mix the two)."""
    m = getattr(x, "date", None)
    return m() if callable(m) else x


def track_record(write: bool = True) -> dict:
    """Score the pre-kickoff-frozen model on every WC2026 match played so far."""
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

    # Freeze the rating on pre-kickoff history only — no look-ahead into tournament results.
    train = [m for m in allm if _as_date(m["date"]) < start]
    dc = DixonColesModel(half_life_days=1100.0)
    dc.fit(train)
    model = MatchModel(dc)

    probs, outs, correct = [], [], 0
    for m in played:
        p = list(model.wdl(m["home_team"], m["away_team"], neutral=True))
        o = _outcome(m["home_score"], m["away_score"])
        probs.append(p)
        outs.append(o)
        correct += int(np.argmax(p) == o)

    s = score_summary(np.array(probs), np.array(outs))
    uni = score_summary(np.array([UNIFORM] * len(outs)), np.array(outs))
    rec.update({
        "status": "live",
        "rps": round(s["rps"], 4),
        "uniform_rps": round(uni["rps"], 4),
        "skill": round(uni["rps"] - s["rps"], 4),
        "log_loss": round(s["log_loss"], 4),
        "brier": round(s["brier"], 4),
        "calls_correct": correct,
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
        print(f"  correct calls {rec['calls_correct']}/{rec['n_matches']}  |  "
              f"log-loss {rec['log_loss']}  brier {rec['brier']}")


if __name__ == "__main__":
    main()
