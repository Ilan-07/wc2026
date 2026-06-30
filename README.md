# World Cup 2026 — Forecast & Intelligence

[![CI](https://github.com/Ilan-07/wc2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Ilan-07/wc2026/actions)
&nbsp;**▶ Live dashboard: [ilan-07.github.io/wc2026](https://ilan-07.github.io/wc2026/)** — auto-updates every matchday (champion odds, projected bracket, predicted match scores, and a live public track record).

A transparent, self-grading, market-anchored forecasting system for the 2026 FIFA World Cup. It runs a full
**hierarchical-Bayesian / Dixon-Coles match model**, simulates the **real tournament draw** by Monte-Carlo,
blends the result with the **betting market**, and **conditions on matches as they are played** so the
forecast sharpens every matchday. Every number is auditable, the uncertainty is reported honestly, and the
system **scores itself in public** against reality.

It is deliberately **not** marketed as a market-beater — the liquid winner market is provably hard to beat
(see [`FINDINGS.md`](FINDINGS.md)). The value is calibration, traceability, and intellectual honesty about
what does and doesn't move a forecast.

---

## Highlights

- 🎯 **Calibrated, backtested** — out-of-sample validated across 9 tournaments (not just fit to one).
- 🔄 **Live & self-updating** — a change-gated daily job re-fits only when new data lands, regenerates the
  dashboard, and publishes it; the page auto-reloads.
- 📊 **Public track record** — every WC2026 match is scored against the **production model** *frozen
  before kickoff* (leakage-free, host advantage + altitude applied per fixture, same builder as the
  forecast), and shown live on the dashboard. Hit-rate is reported as decisive-match accuracy —
  draws are structurally unpickable by a single most-likely outcome (see [`FINDINGS.md`](FINDINGS.md)).
- ⚽ **Predicted match scores** — every fixture gets a scoreline: expected goals, the most-likely exact
  score and its probability, and win/draw/loss. Played matches lock to their real result and knockout
  scores appear once the bracket is set — so the section sharpens live alongside the forecast.
- 🧠 **Bayesian + structural** — hierarchical Bayesian Poisson ratings (PyMC) with MCMC diagnostics, a dynamic
  state-space rating warm-start, an xG blend, host advantage, altitude, fatigue, and a learned shootout model.
- 🔍 **Fully auditable** — a knowledge graph + an Analyst→Market→Contrarian→Judge reasoning pipeline explain
  *why* each team gets its number.
- 🧪 **Honest by construction** — `FINDINGS.md` documents every idea that was tested and **rejected**, not
  just the ones that worked.

## Validation (out-of-sample, proper scoring rules)

| What | Result |
|------|--------|
| **9-tournament W/D/L backtest** (WC 2018/22, Euro 2016/20/24, Copa 2016/19/21/24 — 399 matches) | **RPS 0.195 vs 0.234 uniform → skill +0.039**, positive on **8 of 9** tournaments |
| **Deep-run stage reliability** (reach SF / final / title, 4 World Cups) | Pooled **Brier 0.104, ECE 0.020** |
| **Match-level calibration** | **ECE ~0.03** |
| **Simulation-based calibration** (Talts et al., on the Bayesian sampler) | Parameters recovered cleanly (p = 0.17–0.64) |
| **Model vs market** (honest) | Market wins (RPS 0.190 vs 0.204) → headline blends **25% model / 75% market** |

> The honesty cuts both ways: squad-reputation priors, tournament "DNA", match-importance weighting and an
> XGBoost challenger were all **tested and rejected** by ablation. Full ledger in [`FINDINGS.md`](FINDINGS.md).

## How it works

```
results · odds · xG · squads · injuries
            │
            ▼
   ratings  ─ Dixon-Coles (time-weighted MLE)            ─┐
            ─ Hierarchical Bayesian Poisson (PyMC/MCMC)   │ + xG blend, state-space warm-start
            ─ Dynamic state-space (Kalman) rating         ─┘
            │
            ▼
   Monte-Carlo tournament sim over the REAL draw  ── bootstrap ensemble → honest uncertainty
            │  (host advantage · altitude · fatigue · learned shootout model · injuries)
            ▼
   Market fusion (log opinion pool, market-weighted)  ── Crowd-vs-Model divergence
            │
            ▼
   Dashboard (auto-reloading) + archived JSON + live track record
```

**Live conditioning** locks played group and knockout results so the forecast updates each matchday; the
Round-of-32 bracket — pairs **and tree** — follows the **real published fixtures** the moment the knockout
draw lands: ties and the later-round structure connecting them are read straight from the schedule's
advancement chain, falling back to the official slot template only as a pre-knockout projection.
**Explanation layer** (`intel`): a knowledge graph plus a deterministic Analyst→Market→Contrarian→Judge
pipeline — traceability, not prediction.

## Data sources

| Source | Used for | In forecast? |
|--------|----------|:---:|
| [martj42/international_results](https://github.com/martj42/international_results) | match results & shootouts | ✅ |
| The Odds API | live outright odds | ✅ |
| StatsBomb open data | xG ratings | ✅ |
| Transfermarkt | player market values (key-player weighting) | ✅ |
| Wikipedia | squads, knockout bracket | ✅ |
| Google News RSS / Bluesky | news & social "pulse" | ❌ display-only, fenced from the model |

## Quick start

```bash
pip install -e .                          # core deps; optional extras: bayesian (pymc), intelligence
python fetch_data.py                      # download the open datasets (results, squads, knockout structure)
export ODDS_API_KEY=...                    # optional: live outright odds (The Odds API)
PYTHONPATH=src python cli.py predict       # live forecast + dashboard + archived JSON
open data/processed/wc2026_dashboard.html
```

## CLI

```
python cli.py predict   # live winner forecast + dashboard + archive (refreshes data & odds by default)
python cli.py track     # live tournament track record (scores the forecast vs WC2026 results)
python cli.py intel TEAM# per-team traceable intelligence report (reasoning pipeline + knowledge graph)
python cli.py scenario  # injury / availability what-if
python cli.py score     # grade the model on past World Cups
python cli.py validate  # market-fusion validation on club odds
python cli.py odds      # refresh live outright odds
```

Research / validation gates are also CLI subcommands: `backtest`, `stage-reliability`, `sbc`, `blend-weight`,
`state-space`, `bayes-tau`, `shootout`, `xg-joint`, `draw-pick`, `probe`.

## Automation

`run_daily.sh` + `com.wc2026.daily.plist` (launchd) run the change-gated forecast on a schedule: each tick
pulls fresh results, re-fits **only if a tracked input changed**, regenerates the dashboard, updates the live
track record, and publishes via `publish.sh` to the GitHub-Pages site. Heartbeat ticks are cheap no-ops.

## Project layout

```
src/wc2026/                  # the library
  ratings/{elo,dixon_coles,bayesian_dc,state_space,xg_rating}   model/match_model
  simulate/{format,bracket,tournament}    fusion/{pool,divergence}    graph/kg
  collective/{market,odds_api,sentiment,social,availability}    evaluate/{metrics,score,…}
  intelligence/{squads,injuries,conditions,covariates}    reports/{app,dashboard,scores,explain}    data/{loaders,…}
cli.py  predict.py  score_live.py  intelligence_report.py          # production entry points
*_ablation.py  *_sweep.py  *_validate.py  tournament_backtest.py   # research / validation scripts
run_daily.sh  publish.sh  com.wc2026.daily.plist  tests/  FINDINGS.md
```

The full suite (123 tests) runs locally with `PYTHONPATH=src pytest`. CI lints on every push; a data-fixture
test job is a planned follow-up.

## Scope

An analytics tool, **not betting advice**. The headline champion probability is an uncertain estimate with a
wide, honestly-reported band — you cannot calibrate a single champion number on ~3 tournaments
(see [`FINDINGS.md`](FINDINGS.md)).
