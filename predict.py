"""Live WC2026 winner prediction — the 'working' loop, not a static report.

What makes this different from report.py: it CONDITIONS on matches already played. Re-fetch the
results file, and every completed group match is locked to its real score while only the remaining
matches are simulated — so the champion forecast sharpens after every matchday until one team is
left. Each run is archived (timestamp + data vintage + the full forecast) so it can be scored
against reality afterwards — the thing that makes a forecaster credible.

    PYTHONPATH=src python predict.py            # predict from current data
    PYTHONPATH=src python predict.py --refresh  # re-download results first, then predict

Run it after each matchday during the tournament.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from wc2026.collective import market, sentiment
from wc2026.data import loaders
from wc2026.fusion.pool import pool_two
from wc2026.model.match_model import MatchModel
from wc2026.ratings.dixon_coles import DixonColesModel
from wc2026.ratings.elo import EloModel
from wc2026.simulate.tournament import TournamentSimulator

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ARCHIVE = Path("data/processed/forecasts")
DASHBOARD = Path("data/processed/wc2026_dashboard.html")
# Title-odds blend weight on the model (rest on the de-vigged market). The one validation we have —
# `cli.py validate` on 787 held-out top-5 club matches — learns an optimal model weight of **0.00**:
# against a sharp market the model adds nothing. We can't fit this on internationals (no historical
# WC match-odds set), but international markets are softer than top-5 club books and the international
# model has demonstrated *some* skill vs results (xG blend +0.0058, Bayesian +0.016 RPS). So a small
# positive weight is defensible; 0.35 was optimistic. 0.25 leans hard on the market (the better
# forecaster) while keeping a modest, evidence-backed model voice.
MODEL_WEIGHT = 0.25
NEWS_TEAMS = 14      # teams to pull live headlines for (top by blended odds)


def refresh_results() -> None:
    dest = Path("data/raw/results.csv")
    print("Refreshing results.csv ...")
    subprocess.run(["curl", "-sSL", "--max-time", "40", "-o", str(dest), RESULTS_URL], check=True)


# xG-rating blend weight that won the leave-one-tournament-out gate (+0.0058 RPS); see
# xg_rating_ablation.py. Applied only to the ~50 teams with StatsBomb xG.
XG_BLEND_WEIGHT = 0.4

# Total-goals recalibration (alpha, beta): T' = alpha + beta*(lam+mu), supremacy preserved. Pooled fit
# over 9 tournaments (goals_recalibration.py) — corrects the model's over-dispersed totals (pooled goals
# bias -0.17 -> -0.01). RPS-NEUTRAL on W/D/L by construction; it sharpens scoreline/draw rates only.
GOALS_RECAL = (1.6602, 0.3578)

# --- change-detection state (powers `predict --if-changed`, the frequent-tick path) -----------------
STATE_FILE = Path("data/processed/.last_inputs.json")
# Input files whose bytes define the forecast. results.csv drives all match conditioning (played
# groups, KO results, ratings, vintage); the rest are odds / shootouts / manual locks. Missing files
# are folded into the hash as "absent" so creating one later registers as a change.
_TRACKED_INPUTS = (
    "data/raw/results.csv",
    "data/raw/shootouts.csv",
    "data/raw/wc2026_outright_odds.csv",
    "data/raw/wc2026_injuries.txt",
    "data/raw/wc2026_bracket.txt",
)


def _inputs_fingerprint() -> str:
    """SHA-256 over the bytes of every tracked input file (existence + contents)."""
    import hashlib
    h = hashlib.sha256()
    for rel in _TRACKED_INPUTS:
        p = Path(rel)
        h.update(rel.encode())
        h.update(p.read_bytes() if p.exists() else b"<absent>")
        h.update(b"\0")
    return h.hexdigest()


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _save_state(**updates) -> None:
    state = _load_state()
    state.update(updates)
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _odds_due(min_interval_h: float) -> bool:
    """True if odds were never fetched or the minimum interval has elapsed (budget guard)."""
    last = _load_state().get("last_odds_fetch")
    if not last:
        return True
    try:
        elapsed = (dt.datetime.now() - dt.datetime.fromisoformat(last)).total_seconds()
    except ValueError:
        return True
    return elapsed >= min_interval_h * 3600


def _member(params, xg_blend=None) -> MatchModel:
    m = DixonColesModel(half_life_days=1100.0)
    m.params = params
    return MatchModel(m, xg_blend=xg_blend, goals_recal=GOALS_RECAL)


def _load_xg_blend():
    """Build the (XGRatingParams, weight) blend from cached StatsBomb xG; None if unavailable."""
    import json
    from pathlib import Path

    cache = Path("data/raw/sb_match_records.json")
    if not cache.exists():
        return None
    try:
        from wc2026.ratings.xg_rating import XGRating
        recs = [r for v in json.loads(cache.read_text()).values() for r in v]
        return (XGRating().fit(recs), XG_BLEND_WEIGHT)
    except Exception:
        return None


# Sampler config for the default Bayesian rating. 4 chains (per PyMC's guidance) for clean R-hat.
BAYES_CFG = dict(draws=500, tune=500, chains=4)


# --- Bayesian posterior cache (gap #1: ship the more-accurate rating by default without paying the
# MCMC cost on every rerun). The posterior is cached keyed by data vintage + match count + draw
# count + sampler config, so the first run on a given dataset fits (minutes) and every rerun loads
# instantly; changing the data or the sampler config invalidates the cache automatically. ----
def _bayes_cache_path(matches, n_boot):
    from pathlib import Path
    vintage = max((str(m["date"]) for m in matches), default="none")
    cfg = "-".join(str(BAYES_CFG[k]) for k in ("draws", "tune", "chains"))
    return Path("data/processed/bayesian_cache") / f"{vintage}_{len(matches)}_{n_boot}_{cfg}.json"


def _load_bayes_cache(matches, n_boot):
    import json

    from wc2026.ratings.dixon_coles import DixonColesParams
    p = _bayes_cache_path(matches, n_boot)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return DixonColesParams(**d["base"]), [DixonColesParams(**x) for x in d["draws"]]


def _save_bayes_cache(matches, n_boot, base, draws):
    import json
    from dataclasses import asdict
    p = _bayes_cache_path(matches, n_boot)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"base": asdict(base), "draws": [asdict(d) for d in draws]}))


def build_payload(teams, groups, matches, champ, market_champ, blended, sd, result,
                  avail, n_sims, bayesian, played, news, pick, p_pick) -> dict:
    """Assemble the full data object the interactive dashboard renders from."""
    import hashlib

    import pandas as pd

    from wc2026.data import loaders, venues
    from wc2026.data import squads as squads_mod
    from wc2026.graph.kg import build_kg
    from wc2026.intelligence.injuries import load_manual_availability
    from wc2026.ratings.elo import EloModel
    from wc2026.reports.explain import ReportContext, explain_team
    from wc2026.simulate.bracket import build_official_bracket

    sq = squads_mod.load_squads()
    caps = {t: {p.name: p.caps for p in s.players} for t, s in sq.items()}
    try:  # Transfermarkt market values (quality proxy); degrade to caps if the snapshot is absent
        from wc2026.data.transfermarkt_values import squad_values
        vals = squad_values(sq)
    except Exception:
        vals = {}
    stars = load_manual_availability("data/raw/wc2026_stars.txt")  # editable display list
    df = pd.read_csv(loaders.DEFAULT_RESULTS)
    fv = [(r.home_team, r.away_team, r.city) for r in df.itertuples(index=False)
          if str(r.tournament) == "FIFA World Cup" and str(r.date).startswith("2026")]
    gof = {t: g for g, ts in groups.items() for t in ts}
    alt_cities: dict[str, set] = {}
    for h, a, city in fv:
        if gof.get(h) == gof.get(a):
            v = venues.venue_for_city(str(city))
            if v and v.altitude_m >= 1000:
                alt_cities.setdefault(h, set()).add(v.city)
                alt_cities.setdefault(a, set()).add(v.city)
    kg = build_kg(groups, sq, fv)
    elo = EloModel().fit(matches)
    ctx = ReportContext(teams=teams, elo={t: elo.rating(t) for t in teams}, groups=groups,
                        model_p=champ, market_p=market_champ, blended_p=blended, kg=kg, caps=caps,
                        sd=sd, altitude_cities=alt_cities)

    def hue(t):
        return int(hashlib.md5(t.encode()).hexdigest(), 16) % 360

    team_rows = []
    for t in sorted(teams, key=lambda x: blended[x], reverse=True):
        rep = explain_team(ctx, t)
        c = caps.get(t, {})
        v = vals.get(t, {})
        # key players: curated stars (validated against the squad) first, then backfill by MARKET
        # VALUE (quality) — caps alone surfaced veterans over young stars like Yamal/Pedri. Teams
        # with thin value coverage fall back to most-capped.
        named = [p for p in stars.get(t, []) if p in c]
        if len(named) < 3:
            named += [p for p in sorted(v, key=lambda p: v[p], reverse=True) if p not in named]
        if len(named) < 3:
            named += [p for p in sorted(c, key=lambda p: c[p], reverse=True) if p not in named]
        kp = [f"{p} (€{v[p] / 1e6:.0f}m)" if p in v else f"{p} ({c.get(p, 0)} caps)"
              for p in named[:3]]
        mates = kg.clubmates(t)
        biggest = max(mates.items(), key=lambda kv: len(kv[1]), default=(None, []))
        chem = f"{len(biggest[1])} from {biggest[0]}" if biggest[0] else "dispersed squad"
        ac = alt_cities.get(t, set())
        h = hue(t)
        team_rows.append({
            "team": t, "model": champ[t], "market": market_champ[t], "blended": blended[t],
            "sd": sd[t], "reachSF": result.reach_prob["sf"][t],
            "reachFinal": result.reach_prob["final"][t], "div": market_champ[t] - champ[t],
            "group": gof.get(t, ""),
            "coach": kg.coach(t) or "—", "keyPlayers": kp, "chemistry": chem,
            "altitude": ("plays at " + ", ".join(sorted(ac))) if ac else "no altitude venues",
            "injuries": list(avail.get(t, [])),
            "claims": [{"role": cl.role, "text": cl.text, "evidence": cl.evidence} for cl in rep.claims],
            "g1": f"hsl({h},60%,58%)", "g2": f"hsl({(h + 38) % 360},55%,42%)",
        })

    # projected bracket: group qualifiers fill the official template (only the model gives group
    # standings); knockout advancement uses the BLENDED (market-anchored) title probability, NOT
    # the raw model champ% — the latter double-counts path luck and lets easy-route minnows leapfrog
    # far stronger teams. Strength (blend) is the coherent advancement metric.
    q = result.reach_prob["r16"]
    winners, runners, thirds = {}, {}, []
    for g, ts in groups.items():
        order = sorted(ts, key=lambda t: q[t], reverse=True)
        winners[g], runners[g] = order[0], order[1]
        thirds.append((g, order[2]))
    best = dict(sorted(thirds, key=lambda gt: q[gt[1]], reverse=True)[:8])
    try:
        border = build_official_bracket(winners, runners, best)
    except Exception:
        border = sorted(teams, key=lambda t: blended[t], reverse=True)[:32]
    rounds, alive = [], border
    while len(alive) > 1:
        nd, nxt = [], []
        for i in range(0, len(alive), 2):
            a, b = alive[i], alive[i + 1]
            wa = bool(blended.get(a, 0) >= blended.get(b, 0))
            nd.append({"team": a, "prob": float(blended.get(a, 0)), "win": wa, "group": gof.get(a, "")})
            nd.append({"team": b, "prob": float(blended.get(b, 0)), "win": not wa, "group": gof.get(b, "")})
            nxt.append(a if wa else b)
        rounds.append(nd); alive = nxt

    # groups view: each group's teams ranked by qualification probability
    groups_payload = [{"group": g,
                       "teams": [{"team": t, "qualify": float(q[t])}
                                 for t in sorted(groups[g], key=lambda t: q[t], reverse=True)]}
                      for g in sorted(groups)]

    dv = sorted(teams, key=lambda t: market_champ[t] - champ[t], reverse=True)
    mk = lambda t: {"team": t, "market": market_champ[t], "model": champ[t], "div": market_champ[t] - champ[t]}
    news_payload = [{"team": tp.team, "pulse": tp.pulse, "mood": tp.mood,
                     "items": [{"title": it.title, "source": it.source, "link": it.link} for it in tp.items]}
                    for tp in news.values()]
    # unified "latest" stream across all teams (deduped) — the flowing feed
    latest, seen = [], set()
    for tp in news.values():
        for it in tp.items:
            if it.title and it.title not in seen:
                seen.add(it.title)
                latest.append({"team": tp.team, "title": it.title, "source": it.source,
                               "date": it.date, "link": it.link, "score": it.score})
    vintage = max(m["date"] for m in matches).isoformat()
    mw = int(MODEL_WEIGHT * 100)
    badges = [f"{n_sims:,} simulations", f"blend {mw}/{100 - mw} model/market",
              "Bayesian rating" if bayesian else "MLE rating",
              f"injuries: {sum(len(v) for v in avail.values())} out" if avail else "no injuries applied",
              f"data through {vintage}"]
    return {"kicker": "FIFA World Cup 2026 · Live Forecast", "pick": pick, "pickProb": p_pick,
            "pickSd": sd[pick], "status": f"conditioned on {len(played)}/72 group matches · through {vintage}",
            "badges": badges, "teams": team_rows, "bracket": rounds, "groups": groups_payload,
            "crowdOver": [mk(t) for t in dv[:5]], "crowdUnder": [mk(t) for t in reversed(dv[-5:])],
            "news": news_payload, "latest": latest[:30]}


def main(n_sims: int = 30_000, n_boot: int = 15, refresh: bool = False,
         refresh_odds: bool = False, bayesian: bool = True,
         if_changed: bool = False, reload_secs: int = 300,
         odds_min_interval_h: float = 6.0) -> None:
    fp = None
    if if_changed:
        # Frequent-tick path: always pull the free results feed, fetch paid odds only when the min
        # interval has elapsed, then re-fit ONLY if a tracked input actually changed. This lets the
        # launchd agent run often (live dashboard) without burning CPU or the Odds-API budget.
        try:
            refresh_results()
        except Exception as e:
            print(f"  results refresh failed ({type(e).__name__}: {e}); using cached results.")
        if _odds_due(odds_min_interval_h):
            try:
                from wc2026.collective.odds_api import refresh as refresh_outrights
                print("Refreshing live outright odds (interval due) ...")
                refresh_outrights()
                _save_state(last_odds_fetch=dt.datetime.now().isoformat(timespec="seconds"))
            except Exception as e:
                print(f"  odds refresh failed ({type(e).__name__}: {e}); using cached odds.")
        else:
            print("Odds fetch skipped (within min interval).")
        fp = _inputs_fingerprint()
        if fp == _load_state().get("fingerprint") and DASHBOARD.exists():
            print("no new data — skipping re-fit")
            return
        print("New data detected — recomputing forecast.")
    else:
        if refresh:
            refresh_results()
        if refresh_odds:
            from wc2026.collective.odds_api import refresh as refresh_outrights
            print("Refreshing live outright odds ...")
            refresh_outrights()

    groups = loaders.load_wc2026_groups()
    teams = [t for g in groups.values() for t in g]
    matches = loaders.load_results(since="2014-01-01", min_team_matches=20, keep_teams=set(teams))
    psi = loaders.load_shootout_psi()
    # Learned shootout win-propensity (Lane 2 #3): fit the bias scale from real shootout history
    # instead of the hand-set 0.4. Extra time is already drawn from the DC grid in sample_knockout.
    from wc2026.model.shootout import ShootoutModel
    _so_recs = loaders.load_shootout_records()
    shootout_model = None
    if _so_recs:
        shootout_model = ShootoutModel()
        shootout_model.fit(_so_recs)
        print(f"Shootout model: learned psi-scale {shootout_model.params.psi_scale:+.2f} "
              f"from {len(_so_recs)} historical shootouts.")
    played = loaders.load_wc2026_played_groups()  # conditioning: locked group results
    venue_alt = loaders.load_wc2026_group_venue_altitudes()  # altitude at Mexican venues
    ko_results = loaders.load_wc2026_knockout_results()  # locked knockout results
    known_bracket = loaders.load_bracket_file()  # real R32 bracket once the group stage is done

    stage = (f"knockouts: real bracket + {len(ko_results)} KO results locked"
             if known_bracket else f"{len(played)}/72 group matches played")
    print(f"Conditioning on {stage}.")
    elo = EloModel().fit(matches)
    mm = float(np.mean(list(elo.ratings.values())))
    elo_init = {t: (r - mm) / 400.0 for t, r in elo.ratings.items()}
    # DC warm-start. The dynamic state-space rating (Lane 1 #1) seeds the *non-convex* DC MLE from a
    # better rating than Elo and lands a better optimum out-of-sample (+0.0059 RPS pooled on WC2018+22,
    # positive on both). Used for the MLE path + bootstrap ensemble; the Bayesian default takes no
    # warm-start (it samples from priors) and is unaffected.
    try:
        from wc2026.ratings.state_space import StateSpaceRating
        _ss = StateSpaceRating()
        _ss.fit(matches)
        init = _ss.init_attack()
        print("DC warm-start: dynamic state-space rating (gate +0.0059 RPS vs Elo seed).")
    except Exception as e:
        print(f"  state-space warm-start unavailable ({type(e).__name__}); using Elo seed.")
        init = elo_init
    xg_blend = _load_xg_blend()  # gate-passed StatsBomb xG blend (+0.0058 RPS), None if cache absent
    if bayesian:
        # Hierarchical Bayesian rating — the default: it beat MLE by 0.016 RPS on WC2022 (the biggest
        # single rating gain). MCMC is slow, so the posterior is cached by data vintage — the first
        # run on new data fits (~minutes), reruns load instantly. Falls back to MLE if PyMC is absent.
        cached = _load_bayes_cache(matches, n_boot)
        if cached is not None:
            base_params, draws = cached
            print(f"Loaded cached Bayesian posterior ({len(draws)} draws).")
        else:
            try:
                from wc2026.ratings.bayesian_dc import BayesianDixonColes
                print("Fitting hierarchical Bayesian model (MCMC — a few minutes; cached for reuse)...")
                bdc = BayesianDixonColes(**BAYES_CFG)
                base_params = bdc.fit(matches)
                draws = bdc.posterior_params(n_boot)
                try:
                    print(f"  sampler diagnostics: {bdc.diagnostics()}")
                except Exception:
                    pass
                _save_bayes_cache(matches, n_boot, base_params, draws)
            except Exception as e:
                print(f"  Bayesian rating unavailable ({type(e).__name__}: {e}); using MLE.")
                bayesian = False
        if bayesian:
            ensemble = [_member(p, xg_blend) for p in draws]
    if not bayesian:
        dc = DixonColesModel(half_life_days=1100.0)
        dc.fit(matches, init_attack=init)
        base_params = dc.params
        ensemble = [_member(p, xg_blend)
                    for p in dc.bootstrap(matches, n_boot=n_boot, seed=1, init_attack=init)]
    if xg_blend is not None:
        print(f"xG rating blended (w={XG_BLEND_WEIGHT}) for {len(xg_blend[0].attack)} teams with StatsBomb xG.")
    _ = base_params  # available for diagnostics / missing-team checks

    # Injuries/availability (gap #11) — manual file is the free, immediate path; the API-Football
    # feed is paywalled for the current season. Applies a per-team strength penalty for missing players.
    from wc2026.data import squads as squads_mod
    from wc2026.intelligence.injuries import (
        InjuryAdjustment,
        load_manual_availability,
        penalties_from_scenario,
    )
    avail = load_manual_availability("data/raw/wc2026_injuries.txt")
    if avail:
        _sq = squads_mod.load_squads()
        try:  # value-weight the importance shares (a star out hurts more than a squad filler)
            from wc2026.data.transfermarkt_values import squad_values
            _vals = squad_values(_sq)
        except Exception:
            _vals = {}
        pens = penalties_from_scenario(_sq, avail, values=_vals)
        inj = InjuryAdjustment(pens)
        for m in ensemble:
            m.adjustment = inj
        print(f"Injuries: {sum(len(v) for v in avail.values())} players out across {len(avail)} teams.")

    kw = dict(psi=psi, shootout_model=shootout_model, known_group_results=played,
              group_venue_altitudes=venue_alt, known_bracket=known_bracket,
              known_ko_results=ko_results)
    sim = TournamentSimulator(ensemble, groups, **kw)
    result = sim.run(n_sims=n_sims, seed=0)

    # Per-member spread = parameter uncertainty on the pick.
    per = [TournamentSimulator([m], groups, **kw)
           .run(n_sims=max(3000, n_sims // 8), seed=3).reach_prob["champion"] for m in ensemble]
    sd = {t: float(np.std([p[t] for p in per])) for t in teams}

    champ = result.reach_prob["champion"]  # raw model champion probability (Monte Carlo)

    # Market-anchor the forecast BEFORE naming a winner. Our backtests show the de-vigged market is
    # the better forecaster, so the official pick is the blended number — not the model's standalone
    # favourite, which can be an outlier (e.g. overrating a team with a strong but soft-schedule
    # record and an easy draw). The raw model % is kept for the dashboard's divergence view.
    market_champ = market.devig_outright(market.load_outright_odds(), teams)
    model_row = np.array([[champ[t] for t in teams]])
    market_row = np.array([[market_champ[t] for t in teams]])
    blended = dict(zip(teams, pool_two(model_row, market_row, MODEL_WEIGHT)[0]))

    ranked = sorted(blended.items(), key=lambda x: x[1], reverse=True)  # rank by the blend
    pick, p_pick = ranked[0]

    print("\n" + "=" * 52)
    print(f"  PREDICTED WINNER: {pick}  ({p_pick:.1%} ± {sd[pick]:.1%})")
    print("=" * 52)
    print("\nTop 10 contenders (market-anchored champion probability):")
    for i, (t, p) in enumerate(ranked[:10], 1):
        print(f"  {i:2d}. {t:<13} {p:5.1%}  (model {champ[t]:4.1%}) ± {sd[t]:.1%}")

    top = sorted(teams, key=lambda t: blended[t], reverse=True)[:NEWS_TEAMS]
    print(f"\nFetching live news pulse for: {', '.join(top)}")
    news = sentiment.fetch_many(top, limit=5)
    # Social pulse — DISPLAY ONLY. Attached to the render payload AFTER all model math below; it
    # never reaches champ/blended/ratings (same fence as the news pulse). Inert without a key.
    from wc2026.collective import social
    social_pulse = social.fetch_many(top, limit=5)

    data = build_payload(
        teams, groups, matches, champ, market_champ, blended, sd, result,
        bool(avail) and avail or {}, n_sims, bayesian, played, news, pick, p_pick,
    )
    data["social"] = [  # display-only feed; not consumed by any forecast computation
        {"team": tp.team, "pulse": tp.pulse, "mood": tp.mood,
         "items": [{"title": i.title, "source": i.source, "date": i.date, "link": i.link}
                   for i in tp.items]}
        for tp in social_pulse.values()
    ]
    from score_live import track_record  # live track record (cheap pre-kickoff; writes track_record.json)
    data["track"] = track_record(write=True)
    from wc2026.reports.app import build_app
    html = build_app(data, generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                     reload_secs=reload_secs)
    DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD.write_text(html, encoding="utf-8")
    print(f"Dashboard → {DASHBOARD.resolve()}")

    # --- archive for later scoring ------------------------------------------
    all_matches = loaders.load_results(since="2026-01-01", min_team_matches=0, played_only=True)
    vintage = max((m["date"] for m in all_matches), default=dt.date.today()).isoformat()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "data_vintage": vintage,
        "group_matches_played": len(played),
        "n_sims": n_sims,
        "pick": pick,
        "pick_prob": p_pick,
        "champion_odds": {t: blended[t] for t, _ in ranked[:24]},  # blended = official forecast
        "model_champion_odds": {t: champ[t] for t, _ in ranked[:24]},  # raw model, for divergence
        "uncertainty_sd": {t: sd[t] for t, _ in ranked[:24]},
    }
    out = ARCHIVE / f"forecast_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nArchived → {out}  (data vintage {vintage})")

    if if_changed:  # remember what we just rendered so the next tick can skip an unchanged re-fit
        _save_state(fingerprint=fp or _inputs_fingerprint())


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv, refresh_odds="--refresh-odds" in sys.argv,
         if_changed="--if-changed" in sys.argv)
