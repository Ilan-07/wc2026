"""Live-dashboard tests: change-gated recompute (`predict --if-changed`) + auto-reload tag.

These exercise the cheap gate (fingerprint + odds interval + early-skip) in isolation — no model
fitting and no network — so they stay fast.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

import predict
from wc2026.reports.app import build_app


def _seed_inputs(root: Path) -> None:
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "results.csv").write_text("date,home_team,away_team,home_score,away_score,tournament\n")
    (raw / "wc2026_outright_odds.csv").write_text("# odds\nSpain,5.0\n")


def test_fingerprint_stable_and_sensitive(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_inputs(tmp_path)
    fp1 = predict._inputs_fingerprint()
    assert fp1 == predict._inputs_fingerprint()                       # stable across calls
    (tmp_path / "data" / "raw" / "results.csv").write_text(
        "date,home_team,away_team,home_score,away_score,tournament\n"
        "2026-06-07,Spain,Brazil,2,1,Friendly\n")
    assert predict._inputs_fingerprint() != fp1                       # new content -> new fingerprint


def test_odds_due_respects_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert predict._odds_due(6.0) is True                             # never fetched -> due
    predict._save_state(last_odds_fetch=dt.datetime.now().isoformat(timespec="seconds"))
    assert predict._odds_due(6.0) is False                            # just fetched -> not due
    stale = (dt.datetime.now() - dt.timedelta(hours=7)).isoformat(timespec="seconds")
    predict._save_state(last_odds_fetch=stale)
    assert predict._odds_due(6.0) is True                             # 7h > 6h interval -> due


def test_if_changed_skips_when_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed_inputs(tmp_path)
    # Stub the network side-effects so only the gate logic runs.
    monkeypatch.setattr(predict, "refresh_results", lambda: None)
    monkeypatch.setattr(predict, "_odds_due", lambda *_a, **_k: False)
    # Pre-seed: current fingerprint + an existing dashboard => the skip branch must trigger.
    predict.DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    predict.DASHBOARD.write_text("<html>old</html>")
    predict._save_state(fingerprint=predict._inputs_fingerprint())
    # Reaching the model body would hit this loader — make that an explicit failure.
    monkeypatch.setattr(predict.loaders, "load_wc2026_groups",
                        lambda *a, **k: pytest.fail("re-fit ran despite unchanged inputs"))
    predict.main(if_changed=True)                                     # returns early, no exception
    assert predict.DASHBOARD.read_text() == "<html>old</html>"        # dashboard left untouched


def test_build_app_injects_reload_tag():
    html = build_app({}, generated="2026-06-07 18:00", reload_secs=120)
    assert '<meta http-equiv="refresh" content="120">' in html
    assert "__RELOAD__" not in html                                  # token fully substituted


def test_bracket_order_locks_known_tree_edges():
    # Four R32 ties; a later-round fixture pairs the winners of tie 1 (A) and tie 4 (G), so those
    # two ties must end up as siblings (adjacent blocks) — never folded with their date-neighbours.
    r32 = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    later = [("A", "G")]                                             # real R16 edge: tie1 vs tie4
    out = predict._bracket_order_from_fixtures(r32, later)
    assert sorted(out) == sorted("ABCDEFGH")                         # same 8 teams, no dupes/drops
    ties = [(out[i], out[i + 1]) for i in range(0, 8, 2)]
    assert {frozenset(t) for t in ties} == {frozenset(t) for t in r32}  # every R32 pair preserved
    # round-of-16 folds adjacent ties: ties (0,1) meet, ties (2,3) meet. The A/B and G/H ties must
    # share an r16 slot so A can meet G as the real fixture dictates.
    r16_partners = [frozenset(ties[0]) | frozenset(ties[1]), frozenset(ties[2]) | frozenset(ties[3])]
    assert frozenset("ABGH") in r16_partners


def test_bracket_order_falls_back_chronologically_without_edges():
    # With no later fixtures the ties simply fold in chronological order, two by two.
    r32 = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    out = predict._bracket_order_from_fixtures(r32, [])
    assert out == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_derive_known_bracket_matches_published_r32():
    # Against the live dataset: once the knockout draw is published, the derived bracket must equal
    # the real Round-of-32 ties exactly (32 distinct teams, 16 official pairs) — never the template's
    # re-derivation, which can misallocate third-placed teams.
    fixtures = predict.loaders.load_wc2026_r32_fixtures()
    if not fixtures:
        pytest.skip("knockout draw not yet published in results.csv")
    groups = predict.loaders.load_wc2026_groups()
    played = predict.loaders.load_wc2026_played_groups()
    border = predict._derive_known_bracket(groups, played)
    assert border is not None and len(border) == len(set(border)) == 32
    derived = {frozenset((border[i], border[i + 1])) for i in range(0, 32, 2)}
    assert derived == {frozenset(f) for f in fixtures}
