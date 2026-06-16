"""Live track record tests — pending pre-kickoff, and the scoring path once matches are played.

The tournament hasn't started, so the live path is exercised with injected fixtures + a stubbed
rating so the orchestration and W/D/L scoring are verified without waiting for June 11.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import score_live


def test_outcome_index():
    assert score_live._outcome(2, 1) == 0   # home win
    assert score_live._outcome(1, 1) == 1   # draw
    assert score_live._outcome(0, 2) == 2   # away win


def test_pending_before_kickoff(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(score_live.loaders, "load_wc2026_groups", lambda *a, **k: {"A": ["X", "Y"]})
    monkeypatch.setattr(score_live.loaders, "load_wc2026_group_fixtures",
                        lambda *a, **k: [(date(2026, 6, 11), "X", "Y", "City")])
    monkeypatch.setattr(score_live.loaders, "load_results", lambda *a, **k: [])  # nothing played
    rec = score_live.track_record(write=True)
    assert rec["n_matches"] == 0 and rec["status"] == "pending"
    assert (tmp_path / "data/processed/track_record.json").exists()


def test_live_scoring_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    start = date(2026, 6, 11)
    train = [{"date": date(2024, 1, 1), "importance": "friendly", "home_team": "X", "away_team": "Y",
              "home_score": s % 3, "away_score": (s + 1) % 2} for s in range(30)]
    played = [
        {"date": start, "importance": "world_cup", "home_team": "X", "away_team": "Y",
         "home_score": 2, "away_score": 1},   # home win — model (pick=home) is correct
        {"date": start, "importance": "world_cup", "home_team": "X", "away_team": "Y",
         "home_score": 0, "away_score": 0},   # draw — model is wrong (and unpickable)
    ]
    monkeypatch.setattr(score_live.loaders, "load_wc2026_groups", lambda *a, **k: {"A": ["X", "Y"]})
    monkeypatch.setattr(score_live.loaders, "load_wc2026_group_fixtures",
                        lambda *a, **k: [(start, "X", "Y", "City")])
    monkeypatch.setattr(score_live.loaders, "load_results", lambda *a, **k: train + played)
    monkeypatch.setattr(score_live.loaders, "load_wc2026_group_venue_altitudes", lambda *a, **k: {})
    # Stub the (slow) production model build + the W/D/L it produces so the scoring orchestration is
    # tested deterministically without a real Bayesian fit.
    monkeypatch.setattr(score_live, "_build_model", lambda *a, **k: object())
    monkeypatch.setattr(score_live, "score_fixture",
                        lambda model, h, a, **k: SimpleNamespace(p_home=0.5, p_draw=0.3, p_away=0.2))

    rec = score_live.track_record(write=True)
    assert rec["status"] == "live"
    assert rec["n_matches"] == 2
    assert rec["calls_correct"] == 1          # got the home win, missed the draw
    # The draw is unpickable, so the honest hit-rate is graded on the one decisive match (got it).
    assert rec["n_draws"] == 1
    assert rec["n_decisive"] == 1
    assert rec["calls_decisive"] == 1
    assert 0.0 <= rec["rps"] <= 1.0
    assert rec["skill"] == round(rec["uniform_rps"] - rec["rps"], 4)
