# Reproducibility / evidence audit — `fabecd4`

**Reference:** `main` @ `fabecd4736461a216d974fff8cc8266e6b50f9f9`, treated as immutable. This audit
changes no model code. It adds only `audit/`, on branch `audit/fabecd4`.
**Audited on:** 2026-10-03, macOS, Python 3.12, numpy 2.4.2 / scipy 1.17.0 / pandas 2.3.3 (from `requirements-lock.txt`).
**Audit artifact SHA:** the commit that adds this file. See the note at the end.

## 1. Clean-machine reproduction

The commands are in [`reproduce.sh`](reproduce.sh). They need no API key and take about 5 minutes, plus
about 3.5 minutes for the optional SBC and Bayesian-vs-MLE steps, which need PyMC. In short:

```bash
git checkout fabecd4736461a216d974fff8cc8266e6b50f9f9
pip install numpy==2.4.2 scipy==1.17.0 pandas==2.3.3 && pip install -e . --no-deps
# results.csv + shootouts.csv pinned to martj42/international_results@ff2a795 (2026-06-07), sha256-checked
# 15 football-data.co.uk club-odds CSVs (E0 D1 SP1 I1 F1 × 2122 2223 2324)
PYTHONPATH=src python cli.py backtest | probe | stage-reliability | validate | blend-weight
pip install pymc==6.0.1 arviz==1.1.0 && PYTHONPATH=src python cli.py sbc   # synthetic data, ~3 min
PYTHONPATH=src python bayesian_ablation.py                                  # WC2022 Bayesian vs MLE, ~20 s
```

`reproduce.sh` was executed verbatim in an empty directory on 2026-10-03 (macOS, `python3.12 -m venv` +
`pip`): exit 0, 9 min 21 s including installs, and every value in §4 matched.

The repository's own documented path (`python fetch_data.py` → `cli.py …`) is **not** reproducible as
written. It fetches martj42 `master`, which is unpinned and has changed since the claims were made (§5).
I used `ff2a795` because it was the newest upstream commit before the claims' first commit
(`3411505`, 2026-06-08). With that snapshot, every cheap claim reproduces to the printed precision.

## 2. Data sources: fetched vs committed

| Artifact | Source | Status at `fabecd4` | Used by headline claims? |
|---|---|---|---|
| `data/raw/results.csv`, `shootouts.csv` | martj42/international_results `master` | **Fetched, unpinned, gitignored** | Yes: backtest, probe, stage reliability |
| `data/raw/odds/{E0,D1,SP1,I1,F1}_{2122,2223,2324}.csv` | football-data.co.uk | **Fetched, unversioned URLs, gitignored** | Yes: model vs market, blend weight |
| `sb_*.json` (StatsBomb), Wikipedia squads/knockout JSON, Transfermarkt | GitHub / Wikipedia API / scrape | Fetched, gitignored | No (only the live forecast and xG ablations) |
| Odds API outrights | The Odds API (paid key) | Fetched, gitignored | No (only the live forecast blend) |
| `wc2026_bracket.txt`, `wc2026_injuries.txt`, `wc2026_stars.txt` | hand-maintained | **Committed** | No |
| Any backtest output, posterior, or forecast archive | — | **None committed** (`data/processed/` is gitignored) | — |

No retained evidence exists at `fabecd4`. Every headline number has to be regenerated from fetched data.
The outputs captured for this audit are in `audit/outputs/`.

## 3. Split and time boundaries (leakage)

**9-tournament backtest** (`evaluate/tournament_backtest.py`; unchanged since `3411505`)
- Training set: all matches since 2006-01-01 with `date < kickoff`. The kickoff cutoffs are WC18 2018-06-14,
  WC22 2022-11-20, Euro16 2016-06-10, Euro20 2021-06-11, Euro24 2024-06-14, Copa16 2016-06-03,
  Copa19 2019-06-14, Copa21 2021-06-13 and Copa24 2024-06-20. Elo and Dixon-Coles are refit for each edition. Time decay
  is anchored at the last training match.
- Test set: that edition's matches with `date ≥ kickoff`. All 399 matches are scored (64/64/51/51/51/32/26/28/32);
  none are dropped.
- **No market input** reaches this path, so there is no market leakage here by construction.
- Weak points (not fixed here):
  - (a) The `min_team_matches=15` team filter counts matches over the whole file, *including
    post-cutoff matches*. That means which teams are in each training set depends on future data. This
    is minor, but it is a real look-ahead.
  - (b) The half-life of 1100 days was chosen by `tune_halflife.py` on the WC2018 and WC2022 *test*
    matches. Those 2 of the 9 editions are therefore not hyperparameter-out-of-sample.

**Stage reliability** (`evaluate/stage_reliability.py`)
- Training data starts 2002-01-01 with `date < kickoff` (2010-06-11, 2014-06-12, 2018-06-14,
  2022-11-20). Groups are reconstructed from the actual group fixtures, which were known before the tournament.
- The bracket is a canonical A–H crossover in date order, *not* the real bracket. The script's own
  docstring states this.

**Model vs market / blend weight** (`fusion_validate.py`, `blend_weight_fit.py`)
- For each league, matches are sorted by date. Dixon-Coles is fit on the first 67% (≈ seasons 21/22 and 22/23)
  and tested on the rest. The market is the de-vigged `Avg` pre-match odds.
- The pool weight is fit on a *random* half (`validate`) or with random 5-fold CV (`blend-weight`) inside
  the test period. The folds are not temporal, but only one scalar is fit.
- Scope: this compares the model with the market on **club league matches**, not internationals. The
  README's "25% model / 75% market" is an editorial choice (`predict.py:45`). It is not derived from this
  test, which learns w = 0.00.

**Live WC2026 track record** (`score_live.py`; README: "scored against the production model *frozen before
kickoff* (leakage-free)")
- The data boundary is sound. The model is trained on matches dated before the first WC2026 fixture (2026-06-11),
  and the xG blend uses only xG from before that date.
- **"Frozen" is a data cutoff, not stored predictions.** Every `cli.py track` run refits the model on pre-kickoff
  data using *the code at that moment*. No per-match prediction was recorded before its match.
- **The graded model and the headline metric changed mid-tournament, after results were seen.** On 2026-06-17,
  six days in, `ebe01af` added host advantage and altitude to the graded model, and `3d94bb2` switched the
  headline from raw hit-rate to decisive-match accuracy. FINDINGS gives the motivation as observed WC2026
  outcomes ("8 of the first 16 group matches were draws → a 6/16 raw count that is really 6/8 on decisive
  games"). Each change may be defensible on its own. Even so, the published track record is a cutoff-based
  retrospective whose model and metric were chosen with tournament results in view. It is **not** a frozen
  forecast, and "leakage-free" overstates it.
- What *is* retained: the `gh-pages` branch history holds the published dashboard at each update. The last
  publish before the opening match (`e128e87`, commit time 2026-06-11 17:59 UTC) contains each team's
  pre-tournament title / qualify / reach-SF / reach-final probabilities (model, market and blended). It has no
  per-match W/D/L, and commit timestamps are self-reported, not externally attested. A team-level pre-kickoff
  scorecard could be built from it; a per-match one cannot.

## 4. Claim → artifact table (README "Validation" section)

"Reproduced" means the regenerated number matches the README at the printed precision.

| README claim | Command | Code | Pinned data `ff2a795` | Current upstream `394fe81` | Verdict |
|---|---|---|---|---|---|
| 9-tourn. RPS **0.195** vs uniform **0.234**, skill **+0.039**, **8/9** positive, 399 matches | `cli.py backtest` | `evaluate/tournament_backtest.py` | 0.1951 / 0.2343 / **+0.0392**, 8/9, n=399 | 0.1955 / 0.2343 / **+0.0389**, 8/9 | **Reproduced** (pinned data only for FINDINGS' "+0.0392") |
| Stage reliability pooled **Brier 0.104, ECE 0.020** | `cli.py stage-reliability` | `evaluate/stage_reliability.py` | 0.1044 / **0.020** | 0.1047 / **0.016** | **Reproduced** (pinned data only) |
| Match-level **ECE ~0.03** | `cli.py probe`, "overall … ECE" line | `evaluate/conditional_calibration.py` → `metrics.expected_calibration_error` (10-bin, all classes, the same 399 backtest matches) | **0.031** | 0.028 | **Reproduced**, but the README cites no command. The number is only *asserted* in docstrings, and I identified the source by elimination. |
| Model vs market RPS **0.204 vs 0.190** | `cli.py validate` | `fusion_validate.py` | — | model 0.2043, market 0.1897, n=787 (odds fetched 2026-10-03) | **Reproduced**. The odds are not version-pinned. |
| "→ headline blends 25/75" | — | `predict.py:45` `MODEL_WEIGHT = 0.25` | — | CV weight = **0.00 ± 0.00** (`blend-weight`) | **Not an evidence claim.** It is editorial. `config.py:25` and `blend_weight_fit.py` still say 0.35. |
| SBC "parameters recovered cleanly (p = 0.17–0.64)" | `cli.py sbc` (pymc 6.0.1; 128 replicates on synthetic data, seed 0; 3 min) | `evaluate/sbc.py` | mu0 **0.954**, home **0.026**, sigma_att **0.173**, att_0 **0.637** | (data-independent) | **Reproduced; the README wording is inaccurate.** No output was retained at `fabecd4`, so it was rerun on the unchanged code. The p-values match FINDINGS (home 0.026 exactly). The README range "0.17–0.64" leaves out `home`, which is flagged non-uniform (p = 0.026: a mild upward rank skew, mean rank 283/500 vs 250, and not significant after a Bonferroni correction for 4 tests). It also leaves out mu0 (0.954, a pass). `sbc_validate.py` prints "some parameter flags" and then, in hard-coded text, "is calibrated" regardless of the result. |

**Scope caveat on every row above:** the RPS/Brier/ECE numbers evaluate the **Elo-warm-started
Dixon-Coles MLE** spine. They do not evaluate the model `predict.py` ships by default, which is the
hierarchical Bayesian rating plus the xG blend, goals recalibration and fatigue. The README does not make
this distinction. The evidence that the Bayesian rating beats the MLE comes from FINDINGS, not the README:

| FINDINGS claim | Command | Rerun at `fabecd4`, pinned data | Verdict |
|---|---|---|---|
| Hierarchical Bayesian beats MLE on WC2022: **RPS 0.208 vs 0.224 (−0.016)**, 0 divergences | `bayesian_ablation.py` (16 s) | Bayesian **0.2078**, MLE **0.2241**, 64 matches; 0 divergences, max R-hat **1.014** (PyMC warns > 1.01) | **Reproduced, but the size of the gain is overstated.** The script's MLE baseline uses `since=2014, min_team_matches=20`. The 9-tournament backtest's MLE (`since=2006, min_team_matches=15`) scores **0.2145** on the same 64 WC2022 matches. Against that stronger MLE, the Bayesian gain is **≈0.007, not 0.016**. It rests on one tournament in either case. The script's "minnow shrinkage" printout also shows the Bayesian ratings *less* shrunk than the MLE for all 7 minnows listed, the opposite of its stated expectation. These are non-WC teams and do not affect the RPS. |

Raw logs are in `audit/outputs/results-pinned-ff2a795/`, `results-upstream-394fe81/`, `club-odds-fetched-2026-10-03/`, `sbc-synthetic/` and `bayesian-vs-mle/`.
The pinned SHA-256 hashes are `results.csv` 27d2d19b…7d9992 and `shootouts.csv` cd66ba9c…acc9de. The SHA-256 over the 15 odds-file hashes is 0a2648c1…ecedb1.

## 5. Under-tested fragility: the evidence base is a mutable upstream file

The headline numbers depend on `martj42/international_results@master`. The repo does not pin it, hash it,
or snapshot it, and no test checks a headline value. The backtest test only asserts `0.15 < RPS < 0.25` for WC2018.

This is already observable. Between `ff2a795` (2026-06-07) and `394fe81` (2026-08-26), upstream edited
**pre-2025 history**. It swapped home and away on 1919 China–Philippines rows, inserted three 2005
matches, and relabelled a 2012 "Friendly" as "Philippine Peace Cup". On 2026-06-06 it also renamed
"China PR" to "China".

The result is that the documented reproduction path (`fetch_data.py`) now prints skill **+0.0389**, not +0.0392,
and stage ECE **0.016**, not 0.020. Every team-level Dixon-Coles parameter shifts silently, and a
team rename upstream would drop that team from every training set without any error. The drift is small
today, but nothing in CI or the tests would notice a large one.

A cheap fix, not applied here: pin the upstream commit in `fetch_data.py`, check sha256 hashes, and add a
regression test that runs `backtest` on the pinned snapshot and checks `skill_vs_uniform == 0.0392 ± 1e-4`.

Other gaps noted but not pursued: README says "123 tests", the CI comment says 115, and the suite has
144 `def test_` functions. CI runs lint only, so none of the claims above are checked in CI.

## 6. Audit artifact SHA

The commit on branch `audit/fabecd4` (parent `fabecd4`) that adds `audit/`. Its SHA is reported with
the submission. A file cannot contain its own commit hash.
