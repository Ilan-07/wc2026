"""Tests for the statistical spine: Elo, Dixon-Coles, simulator, and metrics."""

from pathlib import Path

import numpy as np
import pytest

from wc2026.data import loaders, synthetic
from wc2026.evaluate import metrics
from wc2026.model.match_model import MatchModel
from wc2026.ratings import elo
from wc2026.ratings.dixon_coles import DixonColesModel, tau
from wc2026.simulate import format as fmt
from wc2026.simulate.tournament import TournamentSimulator


# --------------------------------------------------------------------- Elo
def test_elo_expected_score_symmetry():
    assert elo.expected_score(1500, 1500) == pytest.approx(0.5)
    assert elo.expected_score(1700, 1500) > 0.5
    # complementary
    assert elo.expected_score(1700, 1500) + elo.expected_score(1500, 1700) == pytest.approx(1.0)


def test_elo_update_is_zero_sum_and_rewards_winner():
    m = elo.EloModel()
    nh, na = m.update("A", "B", 2, 0, importance="world_cup")
    assert nh > 1500 and na < 1500
    assert (nh - 1500) == pytest.approx(-(na - 1500))  # zero sum


def test_goal_difference_multiplier_monotone():
    assert elo.goal_difference_multiplier(1, 0) == 1.0
    assert elo.goal_difference_multiplier(3, 1) == 1.5
    assert elo.goal_difference_multiplier(5, 0) > 1.5


# ----------------------------------------------------------------- Dixon-Coles
def test_tau_special_cases():
    lam, mu, rho = 1.3, 1.1, -0.1
    assert tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)
    assert tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
    assert tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
    assert tau(2, 3, lam, mu, rho) == pytest.approx(1.0)  # no correction elsewhere


def test_score_matrix_is_a_distribution():
    dc = DixonColesModel(max_goals=10)
    teams = synthetic.make_teams(8, seed=0)
    dc.fit(synthetic.generate_history(teams, n_matches=600, seed=3))
    grid = dc.score_matrix(1.5, 1.2)
    assert grid.shape == (11, 11)
    assert grid.sum() == pytest.approx(1.0)
    assert (grid >= 0).all()


def test_mle_recovers_planted_strengths():
    """Fitting the generative process should recover the latent attack ratings (rank order)."""
    teams = synthetic.make_teams(16, seed=0)
    history = synthetic.generate_history(teams, n_matches=6000, seed=5)
    dc = DixonColesModel(half_life_days=None)  # no decay: stationary synthetic data
    params = dc.fit(history)
    true_atk = np.array([teams[t]["attack"] for t in params.teams()])
    fit_atk = np.array([params.attack[t] for t in params.teams()])
    corr = np.corrcoef(true_atk, fit_atk)[0, 1]
    assert corr > 0.85


# ----------------------------------------------------------------- format
def test_rank_group_orders_by_points_then_gd():
    recs = [fmt.TeamRecord(t) for t in ("A", "B", "C", "D")]
    by = {r.team: r for r in recs}
    # A beats everyone, D loses to everyone, B and C split.
    by["A"].add_match("B", 1, 0); by["B"].add_match("A", 0, 1)
    by["A"].add_match("C", 3, 0); by["C"].add_match("A", 0, 3)
    by["A"].add_match("D", 1, 0); by["D"].add_match("A", 0, 1)
    by["B"].add_match("C", 2, 0); by["C"].add_match("B", 0, 2)
    by["B"].add_match("D", 1, 0); by["D"].add_match("B", 0, 1)
    by["C"].add_match("D", 1, 0); by["D"].add_match("C", 0, 1)
    rng = np.random.default_rng(0)
    ranked = [r.team for r in fmt.rank_group(recs, rng)]
    assert ranked[0] == "A"
    assert ranked[-1] == "D"
    assert ranked.index("B") < ranked.index("C")  # B beat C head-to-head & on GD


def test_head_to_head_tiebreak():
    # Three teams level on points & GD; head-to-head decides.
    recs = [fmt.TeamRecord(t) for t in ("A", "B", "C", "D")]
    by = {r.team: r for r in recs}
    # A, B, C all beat D 1-0; among themselves A>B>C in a cycle-free order.
    for w in ("A", "B", "C"):
        by[w].add_match("D", 1, 0); by["D"].add_match(w, 0, 1)
    by["A"].add_match("B", 1, 0); by["B"].add_match("A", 0, 1)
    by["A"].add_match("C", 1, 0); by["C"].add_match("A", 0, 1)
    by["B"].add_match("C", 1, 0); by["C"].add_match("B", 0, 1)
    rng = np.random.default_rng(0)
    ranked = [r.team for r in fmt.rank_group(recs, rng)]
    assert ranked[0] == "A" and ranked[3] == "D"


def test_seeding_order_spreads_top_seeds():
    order = fmt._seeding_order(8)
    assert sorted(order) == list(range(8))
    # seeds 0 and 1 must be in opposite halves
    assert (order.index(0) < 4) != (order.index(1) < 4)


def test_build_bracket_has_32_unique_slots():
    quals = [(f"T{i}", (i % 3, i, 0, 0)) for i in range(32)]
    bracket = fmt.build_bracket_order(quals)
    assert len(bracket) == 32
    assert len(set(bracket)) == 32


# ----------------------------------------------------------------- metrics
def test_rps_perfect_and_worst():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert metrics.ranked_probability_score(probs, np.array([0])) == pytest.approx(0.0)
    # confidently predicting away (idx2) when home (idx0) occurred is maximally wrong
    probs_wrong = np.array([[0.0, 0.0, 1.0]])
    assert metrics.ranked_probability_score(probs_wrong, np.array([0])) == pytest.approx(1.0)


def test_rps_orders_near_miss_below_far_miss():
    home_won = np.array([0])
    pred_draw = metrics.ranked_probability_score(np.array([[0.0, 1.0, 0.0]]), home_won)
    pred_away = metrics.ranked_probability_score(np.array([[0.0, 0.0, 1.0]]), home_won)
    assert pred_draw < pred_away  # ordinal: a draw guess beats an away guess


def test_log_loss_and_brier_basic():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    outcomes = np.array([0, 2])
    assert metrics.log_loss(probs, outcomes) > 0
    assert 0 <= metrics.brier_score(probs, outcomes) <= 2


def test_perfect_calibration_has_low_ece():
    rng = np.random.default_rng(0)
    # generate outcomes consistent with the forecast probabilities
    p_home = rng.uniform(0.2, 0.8, 4000)
    probs = np.column_stack([p_home, np.zeros_like(p_home), 1 - p_home])
    draws = rng.random(4000)
    outcomes = np.where(draws < p_home, 0, 2)
    assert metrics.expected_calibration_error(probs, outcomes) < 0.05


# ----------------------------------------------------------------- simulator
@pytest.fixture(scope="module")
def fitted_model():
    teams = synthetic.make_teams(48, seed=0)
    history = synthetic.generate_history(teams, n_matches=4000, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(history)
    return MatchModel(dc), teams


def test_simulator_probabilities_are_consistent(fitted_model):
    model, teams = fitted_model
    groups = synthetic.make_group_draw(teams, seed=2)
    sim = TournamentSimulator(model, groups)
    res = sim.run(n_sims=2000, seed=0)
    champ = res.reach_prob["champion"]
    # exactly one champion per sim => probabilities sum to 1
    assert sum(champ.values()) == pytest.approx(1.0, abs=1e-9)
    # 32 teams reach R32 each sim => sum of r32 probs == 32
    assert sum(res.reach_prob["r32"].values()) == pytest.approx(32.0, abs=1e-9)
    # "reached final" must dominate "champion" for every team
    for t in teams:
        assert res.reach_prob["final"][t] + 1e-9 >= champ[t]


def test_simulator_is_deterministic_under_seed(fitted_model):
    model, teams = fitted_model
    groups = synthetic.make_group_draw(teams, seed=2)
    sim = TournamentSimulator(model, groups)
    a = sim.run(n_sims=500, seed=42).reach_prob["champion"]
    b = sim.run(n_sims=500, seed=42).reach_prob["champion"]
    assert a == b


# ----------------------------------------------------------------- loaders
_RESULTS_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv"
_has_data = _RESULTS_CSV.exists()


def test_tournament_importance_mapping():
    assert loaders.tournament_importance("FIFA World Cup") == "world_cup"
    assert loaders.tournament_importance("FIFA World Cup qualification") == "qualifier"
    assert loaders.tournament_importance("UEFA Nations League") == "nations_league"
    assert loaders.tournament_importance("UEFA Euro") == "continental"
    assert loaders.tournament_importance("Friendly") == "friendly"


@pytest.mark.skipif(not _has_data, reason="results.csv not downloaded")
def test_wc2026_draw_reconstructs_to_12_groups_of_4():
    groups = loaders.load_wc2026_groups()
    assert len(groups) == 12
    assert all(len(v) == 4 for v in groups.values())
    teams = [t for g in groups.values() for t in g]
    assert len(teams) == len(set(teams)) == 48
    # spot-check a couple of known 2026 placements
    assert "Argentina" in groups["J"]
    assert "Spain" in groups["H"]


_SQUADS = Path(__file__).resolve().parents[1] / "data" / "raw" / "wc2026_squads.json"
_has_squads = _SQUADS.exists()


@pytest.mark.skipif(not _has_squads, reason="wc2026_squads.json not downloaded")
def test_squads_parse_completely():
    from wc2026.data import squads

    sq = squads.load_squads()
    assert len(sq) == 48
    assert sum(len(s.players) for s in sq.values()) > 1200
    assert sq["Argentina"].coach == "Lionel Scaloni"
    messi = next(p for p in sq["Argentina"].players if p.name == "Lionel Messi")
    assert messi.caps > 150 and messi.pos == "FW"


def test_value_weighted_importance_beats_caps_for_young_stars():
    """Market value must rank a young star above a low-value veteran; caps got this backwards."""
    from wc2026.data.squads import Player, Squad
    from wc2026.intelligence.injuries import availability_penalty, player_importance

    star = Player("Young Star", "FW", caps=20, goals=5, age=18.0, club="X", club_nat="ESP")
    vet = Player("Old Veteran", "DF", caps=120, goals=2, age=34.0, club="Y", club_nat="ESP")
    filler = Player("Squad Filler", "MF", caps=5, goals=0, age=25.0, club="Z", club_nat="ESP")
    sq = Squad("Test", players=[star, vet, filler])
    values = {"Young Star": 180e6, "Old Veteran": 8e6}  # filler intentionally unmatched

    caps_imp = player_importance(sq)
    val_imp = player_importance(sq, values)
    assert caps_imp["Old Veteran"] > caps_imp["Young Star"]      # caps reward experience
    assert val_imp["Young Star"] > val_imp["Old Veteran"]        # value rewards quality
    # the unmatched filler is imputed (25th-pct), never zero, so it keeps a small share
    assert 0.0 < val_imp["Squad Filler"] < val_imp["Young Star"]
    # losing the star hurts far more under value weighting than under caps
    assert availability_penalty(sq, ["Young Star"], values=values) > availability_penalty(
        sq, ["Young Star"]
    )


def test_squad_contributions_rank_and_star_reliance():
    from wc2026.data.squads import Player, Squad
    from wc2026.intelligence.injuries import squad_contributions

    star = Player("Star", "FW", caps=20, goals=5, age=20.0, club="X", club_nat="ESP")
    a = Player("A", "MF", caps=30, goals=1, age=27.0, club="Y", club_nat="ESP")
    b = Player("B", "DF", caps=30, goals=0, age=29.0, club="Z", club_nat="ESP")
    sq = Squad("Test", players=[star, a, b])
    c = squad_contributions(sq, {"Star": 180e6, "A": 30e6, "B": 20e6}, top=1)
    assert c["contributions"][0][0] == "Star"          # most valuable ranked first
    assert 0.0 < c["star_reliance"] < 1.0              # single-player share is a valid fraction
    assert c["star_reliance"] == round(c["contributions"][0][1], 3)  # top=1 → just the star's share


_TM_VALUES = Path(__file__).resolve().parents[1] / "data" / "raw" / "transfermarkt" / "latest_market_value.csv"
_has_tm = _has_squads and _TM_VALUES.exists()


@pytest.mark.skipif(not _has_tm, reason="Transfermarkt snapshot not downloaded")
def test_transfermarkt_values_match_real_stars():
    from wc2026.data.squads import load_squads
    from wc2026.data.transfermarkt_values import coverage_report, squad_values

    sq = load_squads()
    vals = squad_values(sq)
    rep = coverage_report(sq, vals)
    assert rep["pct"] > 60  # majority of ~1268 players matched
    # contenders' single most valuable player is their actual modern talisman, not a veteran
    top = lambda t: max(vals[t], key=vals[t].get)  # noqa: E731
    assert top("Spain") == "Lamine Yamal"
    assert top("France") == "Kylian Mbappé"
    assert top("England") == "Jude Bellingham"


def test_xg_rating_recovers_strength_order_and_blend_is_safe():
    from wc2026.ratings.xg_rating import XGRating, blend_rates

    # synthetic xG from known attack/defense: A strong, C weak — rating must recover the order
    truth = {"A": 0.6, "B": 0.0, "C": -0.6}
    mu0 = 0.1
    recs = []
    for h in truth:
        for a in truth:
            if h == a:
                continue
            hx = np.exp(mu0 + truth[h] - (-truth[a]))  # def_j := -atk_j in this symmetric setup
            ax = np.exp(mu0 + truth[a] - (-truth[h]))
            recs.append({"home": h, "away": a, "home_xg": hx, "away_xg": ax})
    p = XGRating(half_life_days=None, reg=0.05).fit(recs)
    assert p.attack["A"] > p.attack["B"] > p.attack["C"]   # strength order recovered
    # blend pulls a covered match toward its xG rate; weight 0 and uncovered are no-ops
    goals = (1.0, 2.0)
    assert blend_rates(goals, p, "A", "C", 0.0) == goals          # w=0 untouched
    assert blend_rates(goals, p, "A", "Nowhere", 0.5) == goals    # uncovered untouched
    bl = blend_rates(goals, p, "A", "C", 0.5)
    assert bl[0] > goals[0]   # A (strong) gets a higher blended scoring rate vs weak C


_SB_RECORDS = Path(__file__).resolve().parents[1] / "data" / "raw" / "sb_match_records.json"


@pytest.mark.skipif(not _SB_RECORDS.exists(), reason="StatsBomb records not downloaded")
def test_xg_rating_ranks_real_teams_sensibly():
    import json

    from wc2026.ratings.xg_rating import XGRating

    recs = [r for v in json.loads(_SB_RECORDS.read_text()).values() for r in v]
    p = XGRating().fit(recs)
    # net xG rating (attack minus conceded) should place elite sides above weak ones
    net = {t: p.attack[t] + p.defense[t] for t in p.teams()}
    assert net["Spain"] > net["Qatar"]
    assert net["France"] > net["Saudi Arabia"]


@pytest.mark.skipif(not _has_squads, reason="wc2026_squads.json not downloaded")
def test_squad_features_separate_elite_from_minnow():
    from wc2026.data import squads
    from wc2026.intelligence.squads import all_squad_features

    feats = all_squad_features(squads.load_squads())
    # top-5-league share: Spain ~ all, Cape Verde ~ none
    assert feats["Spain"]["league_quality"] > feats["Cape Verde"]["league_quality"]
    assert feats["Argentina"]["experience_caps"] > feats["Haiti"]["experience_caps"]
    for f in feats.values():
        assert 0.0 <= f["shared_club_chem"] <= 1.0


@pytest.mark.skipif(not _has_data, reason="results.csv not downloaded")
def test_load_results_shapes_and_recency():
    matches = loaders.load_results(since="2018-01-01", min_team_matches=20)
    assert len(matches) > 1000
    assert all(m["date"].year >= 2018 for m in matches)
    assert {"home_team", "away_team", "home_score", "away_score", "importance"} <= matches[0].keys()


# ----------------------------------------------------------------- fusion
def test_devig_sums_to_one_and_orders_correctly():
    from wc2026.collective.market import devig, overround

    p = devig(2.0, 3.5, 4.0)  # home favourite
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[2]  # shorter odds => higher probability
    assert overround(2.0, 3.5, 4.0) > 0  # bookmaker margin is positive


def test_log_opinion_pool_endpoints():
    from wc2026.fusion.pool import pool_two

    model = np.array([[0.6, 0.25, 0.15]])
    market = np.array([[0.3, 0.3, 0.4]])
    # w=1 -> pure model, w=0 -> pure market (up to renormalisation)
    assert np.allclose(pool_two(model, market, 1.0), model)
    assert np.allclose(pool_two(model, market, 0.0), market)
    mid = pool_two(model, market, 0.5)
    assert mid.sum() == pytest.approx(1.0)
    assert model[0, 0] > mid[0, 0] > market[0, 0]  # geometric mean lies between


def test_fit_model_weight_prefers_better_source():
    from wc2026.fusion.pool import fit_model_weight

    rng = np.random.default_rng(0)
    n = 2000
    outcomes = rng.integers(0, 3, n)
    # market = perfect one-hot, model = uniform noise -> weight should collapse to ~0
    market = np.eye(3)[outcomes] * 0.98 + 0.01
    model = np.full((n, 3), 1 / 3)
    w = fit_model_weight(model, market, outcomes)
    assert w < 0.1


def test_divergence_sign():
    from wc2026.fusion.divergence import divergence

    d = divergence(np.array([[0.5, 0.3, 0.2]]), np.array([[0.7, 0.2, 0.1]]))
    assert d[0, 0] == pytest.approx(0.2)  # market higher on home


# ----------------------------------------------------------------- Phase A core upgrades
def test_bootstrap_returns_ensemble_with_all_teams():
    teams = synthetic.make_teams(10, seed=0)
    hist = synthetic.generate_history(teams, n_matches=600, seed=3)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(hist)
    ensemble = dc.bootstrap(hist, n_boot=4, seed=1)
    names = list(teams)
    assert len(ensemble) == 4
    for p in ensemble:
        assert set(p.attack) == set(names)
    # replicates should differ (genuine perturbation, not copies)
    assert ensemble[0].attack[names[0]] != ensemble[1].attack[names[0]]


def test_importance_weighting_changes_fit():
    teams = synthetic.make_teams(8, seed=0)
    hist = synthetic.generate_history(teams, n_matches=500, seed=3)
    for i, m in enumerate(hist):
        m["importance"] = "friendly" if i % 2 else "world_cup"
    plain = DixonColesModel(half_life_days=None).fit(hist, use_importance=False)
    weighted = DixonColesModel(half_life_days=None).fit(hist, use_importance=True)
    assert plain.attack != weighted.attack  # weighting actually changes the estimates


def test_host_advantage_helps_host_qualify():
    teams = synthetic.make_teams(48, seed=0)
    hist = synthetic.generate_history(teams, n_matches=2500, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(hist)
    dc.params.home = 0.5  # force a clear home effect so the test is not noise-bound
    model = MatchModel(dc)
    groups = synthetic.make_group_draw(teams, seed=2)
    host = list(teams)[5]
    base = TournamentSimulator(model, groups, host_teams=set()).run(n_sims=4000, seed=1)
    hosted = TournamentSimulator(model, groups, host_teams={host}).run(n_sims=4000, seed=1)
    # host advantage in every group game should raise the host's chance of reaching the R16
    assert hosted.reach_prob["r16"][host] > base.reach_prob["r16"][host]


def test_ensemble_simulation_runs():
    teams = synthetic.make_teams(48, seed=0)
    hist = synthetic.generate_history(teams, n_matches=1500, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(hist)
    ensemble = [MatchModel(_params_model(p)) for p in dc.bootstrap(hist, n_boot=3, seed=2)]
    groups = synthetic.make_group_draw(teams, seed=2)
    res = TournamentSimulator(ensemble, groups).run(n_sims=1500, seed=0)
    assert sum(res.reach_prob["champion"].values()) == pytest.approx(1.0, abs=1e-9)


def _params_model(params):
    m = DixonColesModel(half_life_days=None)
    m.params = params
    return m


# ----------------------------------------------------------------- conditions / altitude
def test_altitude_penalty_and_adjustment():
    from wc2026.data.venues import VENUES
    from wc2026.intelligence.conditions import AltitudeAdjustment, altitude_penalty, haversine_km

    azteca = VENUES["Mexico City"]
    sea_level = VENUES["Miami Gardens"]
    # a sea-level team is penalised at altitude, an Andean team is not
    assert altitude_penalty("Canada", azteca) > altitude_penalty("Ecuador", azteca)
    assert altitude_penalty("Ecuador", azteca) == pytest.approx(0.0)
    # no penalty at a sea-level venue
    assert altitude_penalty("Canada", sea_level) == pytest.approx(0.0)
    # adjustment favours the more-adapted side and is antisymmetric
    adj = AltitudeAdjustment(azteca)
    assert adj("Ecuador", "Canada") > 0  # Ecuador (home side) advantaged
    assert adj("Ecuador", "Canada") == pytest.approx(-adj("Canada", "Ecuador"))
    # haversine sanity: a known long hop is a few thousand km
    assert 3000 < haversine_km(49.277, -123.112, 25.958, -80.239) < 6000


def test_simulator_applies_altitude_delta():
    teams = synthetic.make_teams(48, seed=0)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(synthetic.generate_history(teams, n_matches=600, seed=3))
    groups = synthetic.make_group_draw(teams, seed=2)
    sim = TournamentSimulator(
        MatchModel(dc), groups,
        group_venue_altitudes={frozenset(("Ecuador", "Canada")): 2240},
    )
    # Ecuador (Andean) is advantaged over sea-level Canada at altitude; antisymmetric
    assert sim._alt_delta("Ecuador", "Canada") > 0.2
    assert sim._alt_delta("Ecuador", "Canada") == pytest.approx(-sim._alt_delta("Canada", "Ecuador"))
    # no effect where no altitude venue is recorded
    assert sim._alt_delta("Mexico", "Spain") == 0.0


# ----------------------------------------------------------------- injury feed integration
def test_api_football_name_normalization_joins_abbreviated_and_full():
    from wc2026.collective.api_football import normalize

    # the API's "K. Mbappé" must hash to the same key as the squad's "Kylian Mbappé"
    assert normalize("K. Mbappé") == normalize("Kylian Mbappé")
    assert normalize("Vinícius Júnior") == ("junior", "v")
    assert normalize("") == ("", "")


def test_load_manual_availability(tmp_path):
    from wc2026.intelligence.injuries import load_manual_availability

    f = tmp_path / "inj.txt"
    f.write_text("# comment\nFrance: Kylian Mbappé, N'Golo Kanté\nBrazil: Vinícius Júnior\n\n")
    table = load_manual_availability(f)
    assert table == {"France": ["Kylian Mbappé", "N'Golo Kanté"], "Brazil": ["Vinícius Júnior"]}
    assert load_manual_availability(tmp_path / "missing.txt") == {}


# ----------------------------------------------------------------- Phase B injuries
def test_injury_adjustment_is_antisymmetric_and_signed():
    from wc2026.intelligence.injuries import InjuryAdjustment

    adj = InjuryAdjustment({"A": 0.2})  # A is depleted
    # depleted home side -> negative delta (scores less); depleted away -> positive for home
    assert adj("A", "B") == pytest.approx(-0.2)
    assert adj("B", "A") == pytest.approx(0.2)
    assert adj("A", "A") == pytest.approx(0.0)


def test_player_importance_sums_to_one_and_ranks_star_top():
    from types import SimpleNamespace

    from wc2026.intelligence.injuries import player_importance

    squad = SimpleNamespace(players=[
        SimpleNamespace(name="Star", caps=150, goals=60, pos="FW"),
        SimpleNamespace(name="Mid", caps=40, goals=5, pos="MF"),
        SimpleNamespace(name="Sub", caps=3, goals=0, pos="DF"),
    ])
    imp = player_importance(squad)  # type: ignore[arg-type]
    assert sum(imp.values()) == pytest.approx(1.0)
    assert imp["Star"] > imp["Mid"] > imp["Sub"]


def test_injury_lowers_team_title_odds():
    teams = synthetic.make_teams(48, seed=0)
    hist = synthetic.generate_history(teams, n_matches=2500, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(hist)
    from wc2026.intelligence.injuries import InjuryAdjustment

    groups = synthetic.make_group_draw(teams, seed=2)
    victim = list(teams)[5]
    base = TournamentSimulator(MatchModel(dc), groups).run(n_sims=4000, seed=0)
    hurt = TournamentSimulator(
        MatchModel(dc, adjustment=InjuryAdjustment({victim: 0.6})), groups
    ).run(n_sims=4000, seed=0)
    assert hurt.reach_prob["champion"][victim] < base.reach_prob["champion"][victim]


# ----------------------------------------------------------------- live conditioning
def test_conditioning_on_played_results_changes_odds():
    teams = synthetic.make_teams(48, seed=0)
    hist = synthetic.generate_history(teams, n_matches=2500, seed=1)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(hist)
    model = MatchModel(dc)
    groups = synthetic.make_group_draw(teams, seed=2)
    strength = {t: teams[t]["attack"] + teams[t]["defense"] for t in teams}
    strong = max(teams, key=lambda t: strength[t])
    grp = next(g for g, ts in groups.items() if strong in ts)
    # lock the strong team to a 0-3 loss against all three group rivals
    known = {frozenset((strong, o)): (strong, 0, 3) for o in groups[grp] if o != strong}
    base = TournamentSimulator(model, groups).run(n_sims=3000, seed=0)
    cond = TournamentSimulator(model, groups, known_group_results=known).run(n_sims=3000, seed=0)
    # forcing three heavy defeats must cut the team's chance of reaching the R16
    assert cond.reach_prob["r16"][strong] < base.reach_prob["r16"][strong]


# ----------------------------------------------------------------- hierarchical Bayesian
def _has_pymc():
    import importlib.util
    return importlib.util.find_spec("pymc") is not None


@pytest.mark.skipif(not _has_pymc(), reason="pymc not installed")
def test_bayesian_dc_samples_and_shrinks():
    from wc2026.ratings.bayesian_dc import BayesianDixonColes

    teams = synthetic.make_teams(10, seed=0)
    hist = synthetic.generate_history(teams, n_matches=800, seed=5)
    b = BayesianDixonColes(draws=80, tune=80, chains=1, seed=0)
    p = b.fit(hist)
    assert set(p.attack) == set(teams) and p.rho == 0.0
    assert b.diagnostics()["divergences"] == 0
    # partial pooling: fitted attack spread should be finite and centered near 0 (sum-to-zero)
    vals = np.array(list(p.attack.values()))
    assert abs(vals.mean()) < 1e-6 and vals.std() < 2.0
    assert len(b.posterior_params(5)) == 5


# ----------------------------------------------------------------- calibration
def test_reliability_plot_renders(tmp_path):
    from wc2026.evaluate.calibration import reliability_plot

    rng = np.random.default_rng(0)
    preds = rng.uniform(0, 1, 200)
    actual = (rng.random(200) < preds).astype(float)  # well-calibrated by construction
    out = reliability_plot(preds, actual, out=str(tmp_path / "cal.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 0


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "raw" / "results.csv").exists(),
    reason="results.csv not downloaded")
def test_reconstruct_old_wc_groups():
    from wc2026.evaluate.calibration import reconstruct_groups

    g = reconstruct_groups(2018)
    assert len(g) == 8 and all(len(v) == 4 for v in g.values())
    teams = [t for v in g.values() for t in v]
    assert len(set(teams)) == 32


# ----------------------------------------------------------------- registry + leakage
def test_canonical_and_coverage_audit():
    from wc2026.data.teams import audit_coverage, canonical

    assert canonical("USA") == "United States"
    assert canonical("Bosnia & Herzegovina") == "Bosnia and Herzegovina"
    assert canonical("Spain") == "Spain"
    assert audit_coverage({"USA", "Spain"}, {"United States"}) == {"Spain"}
    with pytest.raises(ValueError):
        audit_coverage({"Spain"}, {"United States"}, strict=True)


def test_structural_model_has_no_market_or_sentiment_leakage():
    """Tier A (ratings + match model) must not *import* market/sentiment/fusion — no leakage."""
    import ast
    import inspect

    import wc2026.model.match_model as mm
    import wc2026.ratings.dixon_coles as dc
    import wc2026.ratings.elo as elo

    banned = ("collective", "fusion", "odds_api", "sentiment")
    for mod in (dc, elo, mm):
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
            elif isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            for m in mods:
                assert not any(b in m for b in banned), f"{mod.__name__} imports {m} (leakage)"


# ----------------------------------------------------------------- scoring harness
def test_score_champion():
    from wc2026.evaluate.score import score_champion

    probs = {"Argentina": 0.20, "Spain": 0.15, "Brazil": 0.13}
    s = score_champion(probs, "Spain")
    assert s.p_assigned == pytest.approx(0.15)
    assert s.rank == 2
    assert s.top_pick == "Argentina" and not s.top_pick_correct
    assert s.log_score == pytest.approx(-np.log(0.15))
    # a correct top pick
    assert score_champion(probs, "Argentina").top_pick_correct


def test_score_matches_matches_metrics():
    from wc2026.evaluate.score import score_matches

    probs = np.array([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]])
    out = np.array([0, 2])
    s = score_matches(probs, out)
    assert s["n"] == 2 and s["rps"] > 0 and s["log_loss"] > 0


# ----------------------------------------------------------------- knowledge graph + explain
def test_knowledge_graph_queries():
    from types import SimpleNamespace

    from wc2026.graph.kg import build_kg

    groups = {"A": ["Spain", "Brazil"]}
    P = lambda n, c: SimpleNamespace(name=n, club=c, caps=10, goals=1, pos="MF")
    squads = {
        "Spain": SimpleNamespace(coach="Luis", players=[P("Rodri", "Man City"), P("Yamal", "Barcelona")]),
        "Brazil": SimpleNamespace(coach="Carlo", players=[P("Vini", "Real Madrid"), P("Rapha", "Real Madrid")]),
    }
    kg = build_kg(groups, squads, [("Spain", "Brazil", "Mexico City")])
    assert kg.coach("Spain") == "Luis"
    assert kg.squad("Spain") == {"Rodri", "Yamal"}
    assert "Brazil" in kg.opponents("Spain")
    assert kg.clubmates("Brazil") == {"Real Madrid": ["Rapha", "Vini"]}  # shared-club view
    assert kg.stats()["Player"] == 4


def test_explain_pipeline_roles_and_divergence():
    from wc2026.graph.kg import KnowledgeGraph
    from wc2026.reports.explain import ReportContext, explain_team

    kg = KnowledgeGraph()
    kg.add("Coach X", "COACHES", "A", obj_type="Team")
    for p in ("p1", "p2"):
        kg.add(p, "PLAYS_FOR", "A")
    ctx = ReportContext(
        teams=["A", "B"], elo={"A": 1900, "B": 1600},
        groups={"G": ["A", "B"]}, model_p={"A": 0.20, "B": 0.05},
        market_p={"A": 0.08, "B": 0.05}, blended_p={"A": 0.13, "B": 0.05},
        kg=kg, caps={"A": {"p1": 90, "p2": 40}},
    )
    rep = explain_team(ctx, "A")
    roles = [c.role for c in rep.claims]
    assert roles == ["Analyst", "Market", "Contrarian", "Judge"]
    # model 20% >> market 8% -> the market claim should read as model-rates-higher / value
    assert "value" in next(c.text for c in rep.claims if c.role == "Market")
    assert "p1 (90)" in rep.facts["key players"]


# ----------------------------------------------------------------- StatsBomb xG
def test_teamxg_metrics():
    from wc2026.data.statsbomb import TeamXG

    t = TeamXG("X", matches=3, xg_for=6.0, xg_against=3.0, goals_for=9, goals_against=2)
    assert t.xg_diff_per_match == pytest.approx(1.0)      # (6-3)/3
    assert t.finishing == pytest.approx(1.0)              # (9-6)/3 -> clinical/lucky
    bad = TeamXG("Y", matches=2, xg_for=4.0, goals_for=1)
    assert bad.finishing < 0                              # wasteful/unlucky


# ----------------------------------------------------------------- official bracket
def test_official_bracket_fills_and_respects_third_constraints():
    from wc2026.simulate.bracket import SLOT_SPECS, THIRD_SLOTS, build_official_bracket

    assert len(SLOT_SPECS) == 32 and len(THIRD_SLOTS) == 8
    W = {g: f"W{g}" for g in "ABCDEFGHIJKL"}
    R = {g: f"R{g}" for g in "ABCDEFGHIJKL"}
    thirds = {g: f"3{g}" for g in "ACEGIKBD"}  # 8 qualifying thirds
    br = build_official_bracket(W, R, thirds)
    assert len(br) == 32 and len(set(br)) == 32
    # every third landed in a slot whose candidate groups include its own group
    placed = {i: t for i, t in enumerate(br) if t.startswith("3")}
    for i, team in placed.items():
        assert team[1] in SLOT_SPECS[i][1]  # group letter in the slot's candidate set


def test_official_bracket_matches_fifa_annexe_c_all_495():
    """The engine must reproduce FIFA's Annexe-C third-place allocation for every combination."""
    from wc2026.simulate.bracket import SLOT_SPECS, THIRD_SLOTS, build_official_bracket
    from wc2026.simulate.third_allocation import THIRD_ALLOCATION

    assert len(THIRD_ALLOCATION) == 495  # C(12, 8) — every qualifying-third combination
    # third-slot index -> the group winner it faces in the Round of 32
    winner_of_slot = {i: SLOT_SPECS[i - 1 if i % 2 else i + 1][1] for i in THIRD_SLOTS}
    W = {g: f"W{g}" for g in "ABCDEFGHIJKL"}
    R = {g: f"R{g}" for g in "ABCDEFGHIJKL"}
    for third_groups, official in THIRD_ALLOCATION.items():
        thirds = {g: f"3{g}" for g in third_groups}
        br = build_official_bracket(W, R, thirds)
        assert len(br) == 32 and len(set(br)) == 32
        # winner-group -> third-group as actually built, then compare to the official table
        built = {winner_of_slot[i]: br[i][1] for i in THIRD_SLOTS}
        assert built == official
        # FIFA guarantees no two same-group teams meet in the Round of 32
        for i in range(0, 32, 2):
            assert br[i][1] != br[i + 1][1]


# ----------------------------------------------------------------- knockout conditioning
def test_knockout_conditioning_with_hypothetical_bracket():
    teams = synthetic.make_teams(48, seed=0)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(synthetic.generate_history(teams, n_matches=2000, seed=1))
    groups = synthetic.make_group_draw(teams, seed=2)
    names = list(teams)
    bracket = names[:32]  # supply a real R32 bracket directly
    # lock the opening tie so bracket[1] beats bracket[0]
    locked = {frozenset((bracket[0], bracket[1])): bracket[1]}
    res = TournamentSimulator(
        MatchModel(dc), groups, known_bracket=bracket, known_ko_results=locked
    ).run(n_sims=2000, seed=0)

    # the 32 bracket teams all reach R32; the 16 excluded teams never do (group stage skipped)
    assert all(res.reach_prob["r32"][t] == 1.0 for t in bracket)
    assert all(res.reach_prob["r32"][t] == 0.0 for t in names[32:])
    assert sum(res.reach_prob["champion"].values()) == pytest.approx(1.0, abs=1e-9)
    # the locked result holds in every simulation
    assert res.reach_prob["r16"][bracket[1]] == 1.0   # locked winner always advances
    assert res.reach_prob["r16"][bracket[0]] == 0.0   # locked loser never does


def test_known_bracket_must_be_32():
    teams = synthetic.make_teams(48, seed=0)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(synthetic.generate_history(teams, n_matches=400, seed=1))
    groups = synthetic.make_group_draw(teams, seed=2)
    with pytest.raises(ValueError):
        TournamentSimulator(MatchModel(dc), groups, known_bracket=list(teams)[:30])


# ----------------------------------------------------------------- Tournament DNA
def test_dna_residual_and_adjustment():
    from wc2026.intelligence.tournament_dna import DnaAdjustment, compute_dna

    # A beats B in two World Cup matches despite equal starting ratings -> positive DNA for A.
    matches = [
        {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0, "importance": "world_cup", "neutral": True},
        {"home_team": "B", "away_team": "A", "home_score": 0, "away_score": 1, "importance": "world_cup", "neutral": True},
    ]
    dna = compute_dna(matches, shrink=1.0)
    assert dna["A"] > 0 > dna["B"]
    adj = DnaAdjustment(dna, scale=1.0)
    assert adj("A", "B") == pytest.approx(-adj("B", "A"))


# ----------------------------------------------------------------- P6 market/report
def test_devig_outright_distribution():
    from wc2026.collective.market import devig_outright

    odds = {"A": 5.0, "B": 10.0}
    teams = ["A", "B", "C"]  # C absent -> gets long default
    probs = devig_outright(odds, teams, default_odds=100.0)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["A"] > probs["B"] > probs["C"]


def test_forecast_row_divergence_and_dashboard_html():
    from wc2026.reports.dashboard import ForecastRow, build_dashboard

    rows = [
        ForecastRow("Spain", model=0.07, market=0.14, blended=0.11, reach_sf=0.25, reach_final=0.14),
        ForecastRow("Argentina", model=0.22, market=0.11, blended=0.16, reach_sf=0.44, reach_final=0.31),
    ]
    assert rows[0].divergence == pytest.approx(0.07)  # market higher than model
    html = build_dashboard(rows, news={}, n_sims=1000, model_weight=0.35,
                           generated="2026-06-02 12:00", odds_note="snapshot")
    assert "<table" in html and "Argentina" in html and "News Pulse" in html
    assert "not used in the forecast" in html  # sentiment-is-display-only disclaimer present


def test_headline_sentiment_score():
    from wc2026.collective.sentiment import _headline_score

    assert _headline_score("Star striker returns, confident of a win") > 0
    assert _headline_score("Key defender ruled out with injury, crisis looms") < 0


# ----------------------------------------------------------------- covariates
def test_zscore_and_squad_adjustment_symmetry():
    from wc2026.intelligence.covariates import SquadAdjustment, zscore_features

    feats = {"A": {"x": 2.0, "y": 1.0}, "B": {"x": 0.0, "y": 1.0}, "C": {"x": 1.0, "y": 3.0}}
    names, z = zscore_features(feats)
    assert "y" in names and "x" in names
    # zero-variance features would be dropped; here both vary
    theta = np.array([0.1, 0.0])  # weight only the first standardized feature
    adj = SquadAdjustment(names, z, theta)
    # antisymmetry: delta(A,B) == -delta(B,A)
    assert adj("A", "B") == pytest.approx(-adj("B", "A"))
    # a team adjusted against itself is neutral
    assert adj("A", "A") == pytest.approx(0.0)


def test_match_model_adjustment_shifts_rates():
    teams = synthetic.make_teams(8, seed=0)
    dc = DixonColesModel(half_life_days=None)
    dc.fit(synthetic.generate_history(teams, n_matches=800, seed=3))
    t = dc.params.teams()
    base = MatchModel(dc)
    boosted = MatchModel(dc, adjustment=lambda h, a: 0.3)  # always favour home
    lam0, mu0 = base.rates(t[0], t[1])
    lam1, mu1 = boosted.rates(t[0], t[1])
    assert lam1 > lam0 and mu1 < mu0  # home scores more, concedes less


def test_stronger_teams_win_more_often(fitted_model):
    model, teams = fitted_model
    groups = synthetic.make_group_draw(teams, seed=2)
    sim = TournamentSimulator(model, groups)
    res = sim.run(n_sims=3000, seed=7)
    # team strength score from planted params; top-quartile should out-win bottom-quartile
    strength = {t: teams[t]["attack"] + teams[t]["defense"] for t in teams}
    ranked = sorted(teams, key=lambda t: strength[t], reverse=True)
    champ = res.reach_prob["champion"]
    top = np.mean([champ[t] for t in ranked[:12]])
    bottom = np.mean([champ[t] for t in ranked[-12:]])
    assert top > bottom
