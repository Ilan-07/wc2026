# World Cup 2026 — Forecast

[![CI](https://github.com/Ilan-07/wc2026/actions/workflows/ci.yml/badge.svg)](https://github.com/Ilan-07/wc2026/actions)
&nbsp;**Live dashboard: [ilan-07.github.io/wc2026](https://ilan-07.github.io/wc2026/)** — updated every matchday during the tournament (champion odds, projected bracket, predicted match scores, and a public track record).

A forecasting system for the 2026 FIFA World Cup. It rates teams with a Dixon-Coles / hierarchical-Bayesian
Poisson match model, simulates the real tournament draw by Monte Carlo, blends the result with the betting
market, and conditions on matches as they are played.

It is **not** a market-beater: on the one dataset where both can be scored, the market wins and the fitted
model weight is zero (see [`FINDINGS.md`](FINDINGS.md)). The point of the project is calibration,
reproducibility, and a written record of which ideas helped and which did not.

---

## What it does

- **Forecast.** Rates every team, simulates the tournament over the real draw (host advantage, altitude, rest
  days, a learned penalty-shootout model, injuries), and blends the title odds with the de-vigged market.
- **Live conditioning.** Played group and knockout results are locked in; the knockout bracket follows the
  real published draw.
- **Public track record.** Every WC2026 match is scored by a production model fit only on data from before
  kickoff. This is a **data cutoff, not a stored forecast**. The track record is recomputed with the current
  code each run, and on 2026-06-17 (six days in) the graded model gained host advantage and altitude and the
  headline switched to decisive-match accuracy, partly because of the early draws. Treat it as a
  cutoff-based retrospective, not a pre-registered test. The genuinely frozen evidence is the team-level
  pre-tournament forecast in the `gh-pages` history (last pre-kickoff publish: `e128e87`). See
  [`audit/AUDIT.md`](audit/AUDIT.md) §3.
- **Predicted scores.** Every fixture gets expected goals, the most-likely scoreline, and W/D/L.
- **Research ledger.** [`FINDINGS.md`](FINDINGS.md) lists every factor that was tested, including the ones
  that were **rejected** (squad-reputation priors, tournament "DNA", match-importance weighting, an XGBoost
  challenger, a draw-aware pick).

## Validation

All numbers below are out-of-sample, use proper scoring rules, and are computed on a **pinned,
sha256-verified data snapshot** (see [Reproducing the numbers](#reproducing-the-numbers)). Every row marked
**CI** is re-computed on each push by [`tests/test_reproducibility.py`](tests/test_reproducibility.py),
which fails if a number moves.

| Claim | Result | Model evaluated | Command | Gate |
|---|---|---|---|---|
| **9-tournament W/D/L backtest** (WC 2018/22, Euro 2016/20/24, Copa 2016/19/21/24; 399 matches) | **RPS 0.195 vs 0.234 uniform → skill +0.039**; positive on **8 of 9** (Euro 2016 −0.002) | Elo-seeded Dixon-Coles MLE | `cli.py backtest` | CI |
| **Match-level calibration** (same 399 matches, 10-bin multiclass ECE) | **ECE ≈0.03** (0.030–0.032 by platform) | Elo-seeded Dixon-Coles MLE | `cli.py probe` | CI |
| **Deep-run stage reliability** (reach R16 / QF / SF / final / title; World Cups 2010–22) | Pooled **Brier 0.104, ECE ≈0.02** (0.017–0.020 by platform) | Elo-seeded Dixon-Coles MLE + simulator | `cli.py stage-reliability` | CI |
| **Model vs market** (787 held-out top-5 club-league matches, chronological split, mostly 2023/24) | Market wins: **RPS 0.190 vs 0.204**; CV-fitted model weight **0.00** | Dixon-Coles MLE vs de-vigged odds | `cli.py validate`, `cli.py blend-weight` | CI |
| **Hierarchical Bayesian vs MLE** (WC2022, 64 matches) | Bayesian **0.208** vs the script's MLE 0.224. Against the backtest's (stronger) MLE configuration, which scores 0.2145 on the same matches, the gain is **≈0.007**. **One tournament only.** | Bayesian Poisson (PyMC) | `bayesian_ablation.py` (~20 s) | rerun in `audit/` |
| **Simulation-based calibration** of the Bayesian sampler (Talts et al., 128 replicates) | mu0 / sigma_att / att_0 pass (p = 0.95 / 0.17 / 0.64); **home flagged (p = 0.026)**, not significant after Bonferroni for 4 tests | Bayesian sampler | `cli.py sbc` (~3 min, PyMC) | rerun in `audit/` |

**Numbers vary slightly by machine.** The Dixon-Coles fit (L-BFGS-B on a non-convex likelihood)
stops at slightly different points on different CPU/BLAS builds, and the tournament simulator amplifies
that. Each machine is deterministic, but macOS and two GitHub Linux runner types differ by up to 1×10⁻⁴ in
backtest skill and up to 0.003 in stage ECE (`audit/outputs/cross-platform/observed.md`). The CI tolerances
are set from that measured spread.

**Read this before quoting the table.** The CI-gated rows evaluate the **Elo-seeded Dixon-Coles MLE**
rating. The live forecast's default rating is the **hierarchical Bayesian** model, plus an xG blend, a
total-goals recalibration, and a fatigue term. The evidence that the Bayesian rating beats the MLE is a
single tournament. The headline backtest therefore validates the rating family and the pipeline, **not the
exact model that produced the live numbers**.

**The 25% model / 75% market title-odds blend is an editorial choice, not a fitted value.** The only fit we
can run (club odds) gives the model a weight of 0. International markets are thinner, so a small model weight
is kept. The value lives in `wc2026.config.CONFIG.model_weight`.

**Known limitations of the evidence**
- The half-life (1100 days) was tuned on the WC2018 and WC2022 matches (`tune_halflife.py`). Those two of
  the nine backtest tournaments are therefore not hyperparameter-out-of-sample.
- The established-team filter (`min_team_matches=15`) counts matches over the whole file, including matches
  after each training cutoff. Future data therefore decides which teams, and so which training matches, are
  included, which shifts every team's fitted parameters slightly. It is left as-is so the published numbers
  stay comparable; fixing it is a tracked follow-up.
- Stage reliability uses a canonical A–H bracket, not each edition's real bracket.
- Champion-level calibration cannot be validated on four tournaments.

## Reproducing the numbers

```bash
pip install -r requirements-lock.txt      # or just numpy/scipy/pandas from it; PyMC only for the Bayesian path
pip install -e . --no-deps
PYTHONPATH=src python fetch_data.py       # live feeds -> data/raw/, pinned snapshot -> data/snapshots/ (sha256-checked)
PYTHONPATH=src python cli.py backtest     # ~90 s
PYTHONPATH=src python cli.py probe        # ~80 s
PYTHONPATH=src python cli.py stage-reliability   # ~55 s
PYTHONPATH=src python cli.py validate && PYTHONPATH=src python cli.py blend-weight
```

Validation commands read the snapshot by default: `martj42/international_results` at commit `ff2a795`
(2026-06-07) and 15 football-data.co.uk club-odds seasons, each pinned by hash. The live forecast keeps
reading the moving `master` feed in `data/raw/`. Pass `--live-data` (`cli.py --live-data backtest`) to
evaluate on the live feed. Its numbers drift, because upstream edits historical rows.

## Data sources

| Source | Used for | In forecast? | Validation |
|---|---|:---:|---|
| [martj42/international_results](https://github.com/martj42/international_results) | match results & shootouts | ✅ | pinned @ `ff2a795`, sha256 |
| [football-data.co.uk](https://www.football-data.co.uk) | club odds (model-vs-market validation) | ❌ | sha256 |
| The Odds API | live outright odds | ✅ | — |
| StatsBomb open data | xG ratings | ✅ | — |
| Transfermarkt | player market values (key-player weighting) | ✅ | — |
| Wikipedia | squads, knockout bracket | ✅ | — |
| Google News RSS / Bluesky | news & social "pulse" | ❌ display only, kept out of the model | — |

Nothing above is committed except the hand-maintained `data/raw/*.txt` inputs (stars, injuries, bracket).

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
   Monte-Carlo tournament sim over the REAL draw  ── bootstrap ensemble → uncertainty band
            │  (host advantage · altitude · fatigue · learned shootout model · injuries)
            ▼
   Market fusion (log opinion pool, 25% model / 75% market)  ── model-vs-market divergence
            │
            ▼
   Dashboard + archived JSON + live track record
```

The knockout bracket, both the pairs and the tree, follows the **real published draw**. The R32 ties come
from the results feed. The tree linking them (which tie meets which in the R16, QF and SF) comes from the
published bracket in `data/raw/wc2026_bracket.txt`, because later-round fixtures only enter the feed once
teams advance. If that file is missing, the tree is rebuilt from the schedule's advancement chain as fixtures
appear. Before the knockouts, it falls back to the official slot template.

`cli.py intel TEAM` prints a per-team explanation (knowledge graph plus a rule-based
analyst/market/contrarian summary). It is an **explanation aid only**; it does not change any probability.

## Quick start

```bash
pip install -e .                          # core deps; optional extras: bayesian (pymc), intelligence
PYTHONPATH=src python fetch_data.py       # download the open datasets
export ODDS_API_KEY=...                   # optional: live outright odds (The Odds API)
PYTHONPATH=src python cli.py predict      # live forecast + dashboard + archived JSON
open data/processed/wc2026_dashboard.html
```

## CLI

```
python cli.py predict   # live winner forecast + dashboard + archive (refreshes data & odds by default)
python cli.py track     # live tournament track record (scores the forecast vs WC2026 results)
python cli.py intel TEAM# per-team explanation (knowledge graph + rule-based summary)
python cli.py scenario  # injury / availability what-if
python cli.py score     # grade the model on past World Cups
python cli.py validate  # market-fusion validation on club odds
python cli.py odds      # refresh live outright odds
```

Research and validation gates are also subcommands: `backtest`, `stage-reliability`, `sbc`, `blend-weight`,
`state-space`, `bayes-tau`, `shootout`, `xg-joint`, `draw-pick`, `probe`.

## Automation

`run_daily.sh` with `com.wc2026.daily.plist` (launchd) runs the forecast on a schedule. Each tick pulls fresh
results and re-fits **only if a tracked input changed**. It then regenerates the dashboard, updates the track
record, and publishes to GitHub Pages via `publish.sh`.

## Tests

`PYTHONPATH=src pytest` runs the full suite (148 tests). CI runs lint and the full suite on the pinned
snapshot, including the headline-number regression gate. The PyMC-dependent tests skip in CI.

## Project layout

```
src/wc2026/                  # the library
  ratings/{elo,dixon_coles,bayesian_dc,state_space,xg_rating}   model/match_model
  simulate/{format,bracket,tournament}    fusion/{pool,divergence}    graph/kg
  collective/{market,odds_api,sentiment,social,availability}    evaluate/{metrics,score,…}
  intelligence/{squads,injuries,conditions,covariates}    reports/{app,dashboard,scores,explain}
  data/{loaders,snapshot,…}
cli.py  predict.py  score_live.py  intelligence_report.py          # production entry points
*_ablation.py  *_sweep.py  *_validate.py  tournament_backtest.py   # research / validation scripts
run_daily.sh  publish.sh  com.wc2026.daily.plist  tests/  FINDINGS.md  GAPS.md  audit/
```

## Scope

An analytics tool, **not betting advice**. The headline champion probability is an uncertain estimate with a
wide reported band; a single champion number cannot be calibrated on a handful of tournaments.
