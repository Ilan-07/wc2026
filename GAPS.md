# Gaps & status

A single index of the project's gaps and their state. This is the **operational companion** to
[`FINDINGS.md`](FINDINGS.md): `FINDINGS.md` records *which modelling factors measurably help* (the
research ledger); this file tracks *what is shipped, what is still open, and what is inherent*.

The historical `gap #N` numbering (referenced in code comments and commit history) is **sparse and
non-contiguous** — many numbers were research ablations now folded into `FINDINGS.md`. The table
below lists every `gap #N` still tagged in the source, plus the operational gaps that have no number.
(Note: `tests/test_scores.py` uses its own *local* "gap #" section labels for test grouping; those are
unrelated to this roadmap.)

## Open / actionable

| Gap | What | Impact | Mitigation in place |
|---|---|---|---|
| **Data-feed staleness** | The live forecast is only as fresh as the community `martj42/international_results` GitHub CSV, which lags real-time by days during the tournament. The deciding matchday can be unpublished for 2–3 days, leaving late groups undecided and their Round-of-32 slots projected rather than real. | High during the tournament — the dashboard chronically trails reality. | Two manual overrides exist: hand-append the day's scores to `data/raw/results.csv`, or drop the 32 real R32 teams (bracket order) into `data/raw/wc2026_bracket.txt` — both are now honoured by the bracket view. **No convenient entry tool yet** (proposed: a `cli.py results`/`cli.py bracket` command). |
| **#11 Injuries — live feed** | Availability is one of the few proven-orthogonal signals, but the API-Football injuries/suspensions feed is paywalled for the current season. | Medium — we rely on a manually-maintained file. | `data/raw/wc2026_injuries.txt` (manual) is wired into the forecast; `injury_scenario.py` suggests entries from Wikipedia. The API path (`collective/api_football.py`) is built but inert without a key. |

## Tagged `gap #N` — status

| # | What | Status | Where |
|---|---|---|---|
| 1 | Bayesian posterior cache — ship the more-accurate rating by default without paying the MCMC cost each run | **Shipped** | `predict.py` (posterior cache) |
| 2 | Hierarchical Bayesian rating (partial-pooling Poisson, PyMC) | **Shipped** (`predict --bayesian`); the τ + pooled-per-team-home variant tested **neutral**, not shipped | `ratings/bayesian_dc.py`, `bayesian_ablation.py`, `bayesian_tau_ablation.py` |
| 4 | Dixon-Coles recency half-life tuning | **Closed** — tuned to 1100 days | `tune_halflife.py` |
| 9 | Fatigue / rest-days congestion covariate | **Shipped** (rest-days gated; travel added for WC2026) | `fatigue_ablation.py` |
| 11 | Injuries / availability | **Partial** — manual file shipped; live API paywalled (see Open above) | `collective/api_football.py`, `intelligence/injuries.py` |
| 18 | Refresh ON by default so a stale run can't happen | **Closed** | `cli.py` |
| 20 | Canonical team-name registry + coverage audit | **Closed** | `data/teams.py` |
| 27 | Central configuration — one place for scattered parameters | **Closed** | `config.py` |
| 28 | Reproducible data acquisition — every source in one place | **Closed** | `fetch_data.py` |
| 35 | Champion-level calibration | **Inherent** — cannot be validated on ~3–4 tournaments (see below) | `evaluate/stage_reliability.py`, `evaluate/calibration.py` |

## Research items tested and gated OUT (detail in `FINDINGS.md`)

- **Joint goals+xG rating** — better only on sparse tournament data, not the full-history production DC; the real lever is comprehensive xG history we lack. Stays research.
- **Bayesian τ + pooled per-team home** — neutral (international home advantage barely varies by team). Keep the simpler Poisson.
- **Goals recalibration** — RPS-neutral on W/D/L; shipped only as a scoreline/totals calibration, not an accuracy lever.
- **Draw-aware pick** — worse on accuracy; ships as a documented conservative option, not the default.
- **Squad features / "DNA" / match-importance weighting / XGBoost** — no improvement or worse.

## Inherent limits (physics of the problem, not bugs)

1. You cannot beat the liquid WC-winner betting market (a money-disciplined superset of our info).
2. You cannot validate champion-level calibration on ~3–4 tournaments (gap #35).
3. Football is high-variance: a 20% favourite usually loses. Probabilities, not predictions.

## Recently closed

- **Bracket view used projected qualifiers.** The Round-of-32 was always built from the simulator's
  top-2 qualification probability, which can't tell a group winner from its runner-up (opposite
  halves of the template) and could even rank a real qualifier behind the third-placed team. Now each
  decided group is seeded from its **actual final standings**, the best-8 thirds are ranked by FIFA's
  Annexe-C tiebreakers (points, goal difference, goals scored) once all groups finish, and a
  published `wc2026_bracket.txt` override is used verbatim when present. Projection is kept only for
  groups still in progress.
- **Stale dashboard footer** — the static renderer claimed "host advantage not yet modelled"; host
  advantage *is* modelled (US/Canada/Mexico), so the footer now matches the live dashboard.
