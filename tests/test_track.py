"""Live track record tests — pending pre-kickoff, and the scoring path once matches are played.

The tournament hasn't started, so the live path is exercised with injected fixtures + a stubbed
rating so the orchestration and W/D/L scoring are verified without waiting for June 11.
"""

from __future__ import annotations

from datetime import date

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
    # Enough pre-kickoff X-vs-Y history that the frozen DC fit succeeds (real fit, small data).
    train = [{"date": date(2024, 1, 1), "importance": "friendly", "home_team": "X", "away_team": "Y",
              "home_score": s % 3, "away_score": (s + 1) % 2} for s in range(30)]
    played = [
        {"date": start, "importance": "world_cup", "home_team": "X", "away_team": "Y",
         "home_score": 2, "away_score": 1},   # home win — model (argmax=home) is correct
        {"date": start, "importance": "world_cup", "home_team": "X", "away_team": "Y",
         "home_score": 0, "away_score": 0},   # draw — model is wrong
    ]
    monkeypatch.setattr(score_live.loaders, "load_wc2026_groups", lambda *a, **k: {"A": ["X", "Y"]})
    monkeypatch.setattr(score_live.loaders, "load_wc2026_group_fixtures",
                        lambda *a, **k: [(start, "X", "Y", "City")])
    monkeypatch.setattr(score_live.loaders, "load_results", lambda *a, **k: train + played)
    # Stub only W/D/L so the scoring assertions are deterministic (the DC fit itself is real).
    monkeypatch.setattr(score_live.MatchModel, "wdl", lambda self, h, a, neutral=True: (0.5, 0.3, 0.2))

    rec = score_live.track_record(write=True)
    assert rec["status"] == "live"
    assert rec["n_matches"] == 2
    assert rec["calls_correct"] == 1          # got the home win, missed the draw
    assert 0.0 <= rec["rps"] <= 1.0
    assert rec["skill"] == round(rec["uniform_rps"] - rec["rps"], 4)
