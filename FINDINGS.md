# What actually predicts the World Cup — measured findings

This project's most valuable output is not the forecast; it's a record of which celebrated factors
*measurably* improve prediction and which don't, tested out-of-sample with proper scoring rules
(Ranked Probability Score, lower = better). Every claim below is reproducible from the repo.

## The spine works
- Dixon-Coles correlated-Poisson + Monte Carlo over the real draw.
- Real out-of-sample skill: **WC2018 RPS 0.214, WC2022 RPS 0.215** vs ~0.24 uniform baseline (`score_backtest.py`).
- Well-calibrated at match level (ECE ~0.03). Recency half-life tuned to **1100 days** (`tune_halflife.py`).

## Validation depth (widening the evidence beyond two World Cups)
- **9-tournament backtest** (`tournament_backtest.py`): out-of-sample W/D/L RPS **0.195 vs uniform 0.234,
  skill +0.0392** pooled over WC2018/22 + Euro 2016/20/24 + Copa 2016/19/21/24 (399 matches). Positive on
  **8 of 9** tournaments; the lone miss is the famously chaotic **Euro 2016** (−0.002). Per-tournament RPS
  0.188 ± 0.029 — the spread is the honest measure of how far a single-WC number can move.
- **Stage reliability** (`stage_reliability.py`): full group+bracket sim vs actual deep runs, pooled over the
  four 32-team World Cups 2010–2022. Pooled **Brier 0.104, ECE 0.020**; per-stage ECE shrinks with depth
  (qualify 0.072 → champion 0.018). The deep-run probabilities the headline sells (reach SF/final/title) are
  calibrated, not just the match probabilities. (Mean-pred == base-rate is a mechanical slot-count identity;
  ECE is the real signal.)
- **Simulation-based calibration** (`sbc_validate.py`): SBC (Talts et al.) on the hierarchical-Bayesian
  sampler — 128 prior→data→posterior replicates, rank-uniformity per parameter. mu0/sigma_att/att pass
  cleanly (p = 0.17–0.64); **home** is borderline (p = 0.026, mild upward rank skew) which across 4
  simultaneous tests is within chance and consistent with short-chain tuning, not a broken posterior. The
  inference machinery behind `predict --bayesian` is calibrated on its own generative model.
- **Learned blend weight** (`blend_weight_fit.py`): 5-fold CV of the log-opinion-pool model/market weight on
  ~1,600 held-out league matches → **w = 0.00 ± 0.00** (zero variance across folds); fused == market (RPS
  0.191) ≫ model (0.203). Confirms the market strictly dominates at match level. The report's hand-set 0.35 is
  therefore a **deliberate editorial** weight (keep the model's voice + surface stage probs the market never
  quotes), now made explicit and anchored to the learned value rather than asserted.

## What was tested and FAILED to improve the forecast (the honest part)
| Factor | Test | Verdict |
|---|---|---|
| Aggregate squad features (caps, league, chemistry) | covariate ablation 2018/22 | **no improvement** — redundant with the rating |
| Tournament "DNA" / mentality | covariate ablation 2018/22 | **worse** (−0.004 to −0.010); "top DNA" teams are small-sample noise |
| Match-importance weighting (down-weight friendlies) | ablation 2018/22 | **worse** (−0.006, −0.0045); friendlies carry signal |
| The model vs the betting market | head-to-head RPS | **market wins** (0.190 vs 0.204); fusion weight on model → 0 |
| XGBoost vs Dixon-Coles | same features, 2018/22 | **Dixon-Coles wins** (RPS 0.219 vs 0.231); worse calibration |
| **Total-goals recalibration** (affine map on λ+μ to fix the probe's −0.17 goals bias, supremacy preserved) | leave-one-tournament-out, 9 tournaments (`goals_recalibration.py`) | **neutral on accuracy** — W/D/L RPS 0.1951 → 0.1952 (−0.0000), though it does nearly zero the pooled total-goals bias (−0.169 → −0.011). Total goals is orthogonal to the favourite, and the fitted slope β≈0.4 shows football totals are near-constant (~2.5) and barely predictable. The probe's goals bias is a **calibration curiosity, not an accuracy lever**. **Shipped anyway** (`GOALS_RECAL=(1.66,0.358)` in `predict.py`, `MatchModel(goals_recal=)`) as a totals/scoreline calibration since it's RPS-neutral and zeroes the goals bias; it does not change the W/D/L pick. |
| **Draw-aware pick** (call a draw when `p_draw` clears a learned threshold, instead of pure argmax) | leave-one-tournament-out, 9 tournaments (`draw_pick_calibration.py`) | **worse** — LOTO accuracy 0.529 vs argmax 0.539 (−0.010). The accuracy-maximizing threshold is 0.335 (pooled 0.5414 vs 0.5388), but a draw is almost never any side's single most-likely outcome, so the threshold barely fires and lowering it loses more decisive matches than it buys back draws. **Lesson: hit-rate is the wrong yardstick** — a draw (~26% of matches) is structurally unpickable by a single-outcome call, so the headline "correct calls" now reports **decisive-match accuracy** (draws excluded; `wc2026.evaluate.pick.decisive_accuracy`). The draw-aware `pick` ships as a documented, conservative option at the calibrated 0.335, not as an accuracy lever. (WC2026 illustrates the trap: 8 of the first 16 group matches were draws → a 6/16 raw count that is really **6/8 on decisive games**.) |
| Learned DC rho + **pooled per-team home** on the Bayesian rating | WC2022 vs plain hierarchical Poisson (`bayesian_tau_ablation.py`) | **neutral** — RPS 0.2079 vs 0.2078 (−0.0001). Learned rho is real but tiny (−0.035±0.018); between-team home variance is negligible (σ≈0.05; Bolivia/UAE +0.32 vs Argentina +0.24), so the home hierarchy funnels (home_sigma R-hat 1.4) while forecast params converge (att/def/rho R-hat ≤1.03). International home advantage barely varies by team → pooling adds nothing. Keep the simpler Poisson in production. |

## What passed
| Factor | Test | Verdict |
|---|---|---|
| **Hierarchical Bayesian rating** (partial-pooling Poisson, PyMC) | WC2022 backtest vs MLE | **BETTER** — RPS 0.208 vs 0.224 (−0.016); clean sampling (0 divergences); synthetic recovery corr 0.99 vs 0.85. *The first modeling change to improve the forecast.* (One tournament — thin but positive.) Use `predict --bayesian`. |
| xG as a leading indicator | prior xG-diff vs goal-diff → future result | **xG better** (corr 0.214 vs 0.144) |
| **Penalty-shootout skill (psi)** — learned, leakage-free | temporal backtest on 628 shootouts (`shootout_ablation.py`) | **weak but real** — log-loss 0.6888 vs coin-flip 0.6931 (−0.0043); learned scale **+0.236**, replacing the hand-set 0.4. Penalties are *mostly* a coin flip; the one strong predictor (shoot-first ~60%) is set by a coin toss → unusable in forecasts. Knockout extra time is now sampled from the DC grid (keeps the low-score correlation) instead of independent Poissons. |
| xG as a form covariate *on top of the rating* | ablation | **neutral** — the rating already has it; xG's value needs a full xG-based *rating*, not a bolt-on |
| **Fatigue (rest-days congestion)** | LOTO ablation 2018/22 (`fatigue_ablation.py`) | **BETTER** — pooled RPS −0.0006 (WC2018 −0.0007, WC2022 −0.0004); *positive on both folds*. Small but genuinely orthogonal: a 3-day turnaround is not in the rating. Shipped (rest-days gated; cumulative-travel added for WC2026 as a coord-based prior). |
| **Dynamic state-space rating** (Kalman filter on goal supremacy; latent strength as a random walk, data-tuned drift instead of a hand-set half-life) | leave-one-WC-out sweep 2018/22 (`state_space_sweep.py`) | **BETTER** — out-of-sample RPS **0.2117 vs DC 0.2179 (−0.0062)**; positive on both folds (2018 0.205 vs 0.214, 2022 0.218 vs 0.222). Lets a team's strength evolve and grows its uncertainty while idle, which a single time-weighted snapshot cannot. The drift knob σ_proc/σ_obs is tuned by RPS, not assumed. **Shipped** as the DC MLE warm-start (see below). |

## Shipped to the live forecast (`predict.py`)
Two Lane-1/2 results are wired into production; the rest stay as gated research (a layer ships only when it beats the incumbent **in the production pipeline**, not just in isolation):
- **Dynamic state-space rating → DC warm-start.** The DC log-likelihood is non-convex, so the seed matters: re-seeding the MLE from the state-space rating (instead of Elo) lands a better optimum out-of-sample — **+0.0059 RPS pooled on WC2018+2022, positive on both**. Used for the MLE rating + bootstrap ensemble; the default hierarchical-Bayesian path takes no warm-start (it samples from priors) and is unchanged.
- **Learned shootout model + DC-grid extra time.** `predict.py` fits `ShootoutModel` on 677 real shootouts (learned scale **0.236**, replacing the hand-set 0.4) and passes it to the simulator; knockout extra time is now drawn from the DC grid (keeps the low-score correlation).
- **Total-goals recalibration** (`GOALS_RECAL=(1.66,0.358)`). Affine map on λ+μ (supremacy preserved) that zeroes the model's over-dispersed-totals bias. RPS-neutral on W/D/L — it calibrates scorelines/draw rates, not the pick. (Minor knock-on: slightly compresses goal difference, a group tiebreaker — verified the champion odds don't move materially.)
- **Not wired (honest):** the *joint goals+xG rating* (#4) beats goals-only only on sparse tournament data, not the full-history DC + existing xG blend already in production, and the earliest WC has no prior open xG — so it stays research. *Bayesian τ+pooled-home* (#2) was a dead heat → keep the simpler Poisson. The validation harnesses (#5–#8) ship as `cli.py` subcommands.
| **Joint goals+xG rating** (measurement-error: one latent attack/defense fit to goals-Poisson **and** xG-Gaussian channels at once, vs the post-hoc geometric blend) | leave-one-tournament-out, 4 StatsBomb comps (`xg_joint_ablation.py`) | **BETTER** — RPS **0.2095 vs goals-only 0.2387 (−0.0293)** on 132 xG-covered held-out matches; gain grows monotonically with the xG weight. Confirms xG is the better *measurement* of the same rate when goals are sparse/noisy, and the joint fit is far stronger than the post-hoc blend (+0.0058). **Caveat:** beats a goals-only rating *on the same ~4-tournament data*, not the full-history production DC (which averages goal noise over ~49k matches); the real lever is comprehensive xG history we lack. |

## The unifying lesson
A layer only helps if it carries information **not already in match results**. Squad reputation, DNA,
and FIFA rank are mirrors of strength the rating already captured → noise. xG *denoises* results → the
one signal with merit, but only if it rebuilds the rating (needs comprehensive xG data we lack). The
genuinely orthogonal signals are the ones carrying information **not in past results**: **fatigue /
rest-days** (now confirmed to help, +0.0006), **injuries/availability**, and **market movement**.
Everything else — knowledge graph, agents, sentiment — is valid for *explanation and usability*, not accuracy.

## Inherent limits (not bugs — physics of the problem)
1. You cannot beat the liquid WC-winner betting market (it's a money-disciplined superset of our info).
2. You cannot validate champion-level calibration on ~3 tournaments.
3. Football is high-variance: a 20% favourite usually loses. Probabilities, not predictions.
