# Reproducibility & Evidence Audit — `wc2026` @ `fabecd4`

| | |
|---|---|
| **Reference (immutable)** | [`fabecd4736461a216d974fff8cc8266e6b50f9f9`][ref], the `main` head specified for this gate |
| **Scope** | Headline RPS / Brier / ECE claims in the README Validation table ([`README.md:39–46`][readme-val]), their data, and their leakage boundaries |
| **Out of scope** | Later commits to `main`, the follow-up branch `fix/reproducibility`, and any model improvement |
| **Audit artifact** | One fixed commit SHA, stated in the submission (see [§6](#6-audit-artifact-sha)) |
| **Environment** | macOS + Linux (GitHub Actions, AMD EPYC 7763) · Python 3.12 · numpy 2.4.2 · scipy 1.17.0 · pandas 2.3.3 · pymc 6.0.1 |
| **Audited** | 2026-10-03 |

> [!IMPORTANT]
> Every quote, path, line number and statement about tests or CI below refers to the tree at `fabecd4` and
> nothing later. All code links are permalinks to that commit. If `main` moves, this audit does not move
> with it. No model code was changed; this audit adds only `audit/`.

---

## Summary

**All five testable headline claims reproduce from a clean checkout in about 8 minutes**, but only once
the upstream results feed is pinned. Three of the five need a qualification, and the sixth (the blend
weight) is an editorial choice rather than a measured result.

| # | Headline claim (README @ `fabecd4`) | Reproduced value | Verdict |
|:-:|---|---|---|
| 1 | 9-tournament backtest: RPS **0.195** vs uniform **0.234**, skill **+0.039**, 8/9 positive | 0.1951 / 0.2343 / +0.0392, 8/9, n = 399 | ✅ Reproduced |
| 2 | Stage reliability: pooled **Brier 0.104, ECE 0.020** | 0.1044 / 0.020 (macOS) · 0.1043 / 0.019 (Linux) | ⚠️ Reproduced; ECE is platform-sensitive |
| 3 | Match-level calibration: **ECE ~0.03** | 0.031 | ⚠️ Reproduced; README cites no command (source found by elimination) |
| 4 | Model vs market: **RPS 0.204 vs 0.190** | 0.2043 vs 0.1897, n = 787 | ✅ Reproduced |
| 5 | SBC: "parameters recovered cleanly (**p = 0.17–0.64**)" | mu0 0.954 · **home 0.026** · sigma_att 0.173 · att_0 0.637 | ⚠️ Reproduced; README omits the flagged `home` result |
| 6 | Headline blend **25% model / 75% market** | Fitted model weight **0.00 ± 0.00** | ➖ Editorial choice, not an evidence claim |

✅ matches at printed precision · ⚠️ matches, with a material qualification · ➖ not testable as stated

**Key findings**

1. **Fragility: the evidence rests on a mutable upstream file** ([§5](#5-under-tested-fragility)). Upstream
   has since edited historical rows. The repo's own documented path now gives skill +0.0389 instead of
   +0.0392, and no test notices.
2. **Scope gap.** All headline numbers evaluate the Elo-seeded **Dixon-Coles MLE**, not the hierarchical
   Bayesian model that `predict.py` ships by default. The README does not say so.
3. **The Bayesian gain is overstated.** It reproduces (0.2078 vs 0.2241, WC2022), but against the backtest's
   stronger MLE configuration (0.2145) the gain is **≈0.007, not 0.016**, on one tournament.
4. **The live track record is not a frozen forecast.** It is a pre-kickoff *data cutoff*, recomputed with
   current code. The graded model and headline metric were changed six days into the tournament
   ([§3](#3-split-and-time-boundaries)).

---

## 1. Clean-machine reproduction

[`reproduce.sh`](reproduce.sh) needs no API keys and runs in **~5 min**, plus **~3.5 min** for the optional
PyMC steps.

```bash
REF=fabecd4736461a216d974fff8cc8266e6b50f9f9
git clone https://github.com/Ilan-07/wc2026 && cd wc2026
git checkout --detach "$REF" && test "$(git rev-parse HEAD)" = "$REF"   # refuses to run on any other commit
pip install numpy==2.4.2 scipy==1.17.0 pandas==2.3.3 && pip install -e . --no-deps
# data: results/shootouts @ martj42 ff2a795 + 15 club-odds files, all sha256-verified (see §2)
PYTHONPATH=src python cli.py backtest            # claim 1
PYTHONPATH=src python cli.py probe               # claim 3 ("overall … ECE")
PYTHONPATH=src python cli.py stage-reliability   # claim 2
PYTHONPATH=src python cli.py validate            # claim 4
PYTHONPATH=src python cli.py blend-weight        # claim 6
pip install pymc==6.0.1 arviz==1.1.0
PYTHONPATH=src python cli.py sbc                 # claim 5 (synthetic, ~3 min)
PYTHONPATH=src python bayesian_ablation.py       # finding 3 (~20 s)
```

| Verification run | Result |
|---|---|
| `reproduce.sh` executed verbatim in an empty directory (macOS) | exit 0 · 7 min 45 s incl. installs · HEAD = `fabecd4`, clean tree · all values match |
| Unmodified `fabecd4` on Linux, 3 runs ([harness][harness], run [37105232750][run]) | SHA verified each run · identical across runs · values in [§4](#4-claim-to-artifact-table) |

> [!NOTE]
> The repository's own documented path (`python fetch_data.py`, then `cli.py …`) does **not** reproduce
> the published numbers, because it fetches upstream `master`. The pin `ff2a795` is the newest upstream commit
> before the claims were first committed (`3411505`, 2026-06-08).

---

## 2. Data sources: fetched vs committed

| Artifact | Source | Status at `fabecd4` | Feeds claims |
|---|---|---|:-:|
| `results.csv`, `shootouts.csv` | martj42/international_results `master` | Fetched · **unpinned** · gitignored | 1, 2, 3 |
| `odds/{E0,D1,SP1,I1,F1}_{2122,2223,2324}.csv` | football-data.co.uk | Fetched · **unversioned URLs** · gitignored | 4, 6 |
| StatsBomb, Wikipedia, Transfermarkt | GitHub / Wikipedia API / web | Fetched · gitignored | — (live forecast only) |
| Odds API outrights | The Odds API (paid key) | Fetched · gitignored | — (live forecast only) |
| `wc2026_{bracket,injuries,stars}.txt` | hand-maintained | **Committed** | — |
| Backtest outputs, posteriors, forecast archives | — | **None committed** (`data/processed/` gitignored) | — |

**At `fabecd4` no evidence is retained.** Every headline number must be regenerated. This audit's captured
outputs are indexed in the [appendix](#appendix-evidence-index).

**Pinned inputs used by this audit**

| Input | Pin | SHA-256 |
|---|---|---|
| `results.csv` | martj42 `ff2a795` (2026-06-07) | `27d2d19b…7d9992` |
| `shootouts.csv` | martj42 `ff2a795` | `cd66ba9c…acc9de` |
| 15 club-odds files | content hash (completed seasons) | hash of the 15 hashes: `0a2648c1…ecedb1` |

---

## 3. Split and time boundaries

| Evaluation | Train | Test | Market input | Weaknesses (not fixed) |
|---|---|---|---|---|
| **9-tournament backtest** ([code][tb]) | All matches since 2006-01-01 with `date < kickoff`; Elo → Dixon-Coles refit per edition; time decay anchored at last training match | That edition's matches, `date ≥ kickoff`; all 399 scored | **None** | (a) team filter looks ahead; (b) half-life tuned on 2 of the 9 test editions |
| **Stage reliability** ([code][sr]) | Since 2002-01-01, `date < kickoff` (WC 2010 / 14 / 18 / 22) | Who actually reached R16 / QF / SF / final / title | None | Canonical A–H bracket, not the real one (documented in code) |
| **Model vs market** (`fusion_validate.py`) | Per league, first 67% of date-sorted matches (≈ 21/22–22/23) | Remaining matches; market = de-vigged `Avg` pre-match odds | Benchmark only | Pool-weight folds are random, not temporal (one scalar) |

<details>
<summary><b>Backtest cutoffs per edition</b> (macOS values; Linux differs only in the 4th decimal)</summary>

| Edition | Train cutoff (kickoff) | n | RPS | Uniform | Skill |
|---|---|--:|--:|--:|--:|
| World Cup 2018 | 2018-06-14 | 64 | 0.2143 | 0.2439 | +0.0296 |
| World Cup 2022 | 2022-11-20 | 64 | 0.2145 | 0.2387 | +0.0242 |
| Euro 2016 | 2016-06-10 | 51 | 0.2336 | 0.2320 | −0.0016 |
| Euro 2020 | 2021-06-11 | 51 | 0.1862 | 0.2386 | +0.0524 |
| Euro 2024 | 2024-06-14 | 51 | 0.1876 | 0.2222 | +0.0346 |
| Copa América 2016 | 2016-06-03 | 32 | 0.1881 | 0.2465 | +0.0584 |
| Copa América 2019 | 2019-06-14 | 26 | 0.1512 | 0.2201 | +0.0689 |
| Copa América 2021 | 2021-06-13 | 28 | 0.1558 | 0.2242 | +0.0684 |
| Copa América 2024 | 2024-06-20 | 32 | 0.1602 | 0.2309 | +0.0707 |
| **Pooled** | | **399** | **0.1951** | **0.2343** | **+0.0392** |

</details>

**Leakage weaknesses in detail**

- **(a) Team-filter look-ahead.** `min_team_matches=15` ([`loaders.py:91`][l91]) counts matches over the
  whole file, including post-cutoff ones. Future data therefore decides which teams, and so which training
  matches, are included, which shifts every fitted parameter slightly.
- **(b) Hyperparameter reuse.** The 1100-day half-life was chosen by `tune_halflife.py` on the WC2018 and
  WC2022 test matches, so those two editions are not hyperparameter-out-of-sample.
- **Blend weight.** The README's 25/75 blend ([`README.md:45`][r45], [`predict.py:45`][p45]) is not derived
  from the market test, which learns w = 0.00. The value also disagrees across files: 0.35 in
  [`config.py:25`][c25], [`blend_weight_fit.py:18`][b18] and `report.py:33`.

**Live WC2026 track record** ([`score_live.py`][sl]; [`README.md:23`][r23]: "*frozen before kickoff*
(leakage-free)")

| Aspect | Finding |
|---|---|
| Data boundary | ✅ Sound: training uses matches before the first WC2026 fixture (2026-06-11), and xG before that date |
| "Frozen" | ⚠️ A data cutoff, not stored predictions. Each `cli.py track` run refits with the code of that moment |
| Changes after results were seen | ⚠️ On 2026-06-17, `ebe01af` added host advantage and altitude to the graded model, and `3d94bb2` switched the headline to decisive-match accuracy. [`FINDINGS.md:42`][f42] cites the early draws (8 of 16) as the motivation |
| Retained pre-kickoff evidence | `gh-pages` commit `e128e87` (2026-06-11 17:59 UTC): team-level title / qualify / SF / final probabilities. **No per-match W/D/L**; timestamps are self-reported |
| Verdict | A cutoff-based retrospective, not a pre-registered forecast. "Leakage-free" overstates it |

---

## 4. Claim-to-artifact table

"Reproduced" means the unmodified `fabecd4` code on the pinned data matches the README at printed precision.

| # | Claim | Command | Code | Pinned `ff2a795` | Upstream today `394fe81` |
|:-:|---|---|---|---|---|
| 1 | RPS 0.195 / 0.234, skill +0.039, 8/9 | `cli.py backtest` | [`tournament_backtest.py`][tb] | 0.1951 / 0.2343 / **+0.0392** · 8/9 | 0.1955 / 0.2343 / **+0.0389** · 8/9 |
| 2 | Brier 0.104, ECE 0.020 | `cli.py stage-reliability` | [`stage_reliability.py`][sr] | 0.1044 / **0.020** | 0.1047 / **0.016** |
| 3 | Match ECE ~0.03 | `cli.py probe` | `conditional_calibration.py` → `metrics.expected_calibration_error` (10-bin, all classes, same 399 matches) | **0.031** | 0.028 |
| 4 | Model 0.204 vs market 0.190 | `cli.py validate` | `fusion_validate.py` | 0.2043 vs 0.1897 · n = 787 | (odds not versioned) |
| 5 | SBC p = 0.17–0.64 | `cli.py sbc` | `evaluate/sbc.py` (128 replicates, seed 0) | 0.954 / **0.026** / 0.173 / 0.637 | (synthetic data) |
| 6 | Blend 25/75 | `cli.py blend-weight` | [`predict.py:45`][p45] | CV weight 0.00 ± 0.00 | — |

**Additional claim (`FINDINGS.md`): hierarchical Bayesian beats MLE**

| Claim | Command | Rerun at `fabecd4` | Assessment |
|---|---|---|---|
| WC2022 RPS 0.208 vs 0.224 (−0.016), 0 divergences | `bayesian_ablation.py` (16 s) | 0.2078 vs 0.2241 · 0 divergences · max R-hat 1.014 | Reproduced, but the baseline is weak. The script's MLE uses `since=2014, min 20`; the backtest's MLE (`since=2006, min 15`) scores **0.2145** on the same 64 matches → gain **≈0.007**. One tournament only |

<details>
<summary><b>Notes on claims 3 and 5, and on the Bayesian ablation</b></summary>

- **Claim 3.** The README cites no command for "ECE ~0.03"; the number is only asserted in docstrings. The
  `probe` "overall" line is the only computation that produces it.
- **Claim 5.** No SBC output was retained at `fabecd4`, so it was rerun on the unchanged code; the p-values
  match FINDINGS exactly. `home` (p = 0.026; mean rank 283/500 vs 250) is flagged non-uniform but is not
  significant after Bonferroni correction for 4 tests (0.0125). `sbc_validate.py` prints "some parameter
  flags" and then, in hard-coded text, "is calibrated" regardless of the result ([`sbc_validate.py:36`][s36]).
- **Bayesian ablation.** Its "minnow shrinkage" printout shows the Bayesian ratings *less* shrunk than the
  MLE for all 7 minnows listed, the opposite of its stated expectation. These are non-WC teams, so they do
  not affect the RPS.

</details>

**Scope caveat.** Claims 1–4 evaluate the **Elo-seeded Dixon-Coles MLE**. They do not evaluate the model
`predict.py` ships by default (hierarchical Bayesian + xG blend + goals recalibration + fatigue).

### Cross-platform variation

The same unmodified `fabecd4` code on the same pinned data:

| Platform | Skill | Pooled RPS | Match ECE | Stage Brier | Stage ECE | SF-stage ECE |
|---|--:|--:|--:|--:|--:|--:|
| macOS (3 runs, identical) | +0.0392 | 0.1951 | 0.031 | 0.1044 | **0.020** | 0.039 |
| Linux AMD EPYC 7763 (3 runs, identical) | +0.0392 | 0.1951 | 0.031 | 0.1043 | **0.019** | 0.024 |

**Cause.** The Dixon-Coles fit ([`dixon_coles.py:190`][dc190]) uses `L-BFGS-B` with a finite-difference
gradient and default tolerances on a non-convex likelihood, so it stops at slightly different points across
floating-point builds. The stage simulator amplifies this: knockouts use 1–3 random draws per game, so one
flipped result desynchronises the rest of the seeded stream.

**Consequences.**
- Claims 1, 3 and 4, and the Brier in claim 2, hold on both platforms. "Stage ECE 0.020" is a macOS value.
- FINDINGS' "per-stage ECE shrinks with depth" ([`FINDINGS.md:18`][f18]) does not hold. It is not
  monotone even on macOS (0.072 / 0.034 / 0.039 / 0.032 / 0.018), and the SF value moves from 0.039 to 0.024.
- A wider spread (skill +0.0391, stage ECE 0.017) appeared on another Linux runner type, but in CI runs of
  the follow-up branch, not `fabecd4`'s own tree. It is recorded as corroboration only
  ([`observed.md`](outputs/cross-platform/observed.md)).

---

## 5. Under-tested fragility

> [!WARNING]
> **The evidence base is a mutable upstream file, and nothing tests the headline values.** The repo neither
> pins, hashes nor snapshots `martj42/international_results@master`. The only backtest assertion is
> `0.15 < RPS < 0.25` for WC2018 ([`test_lane3.py:41`][t41]), and at `fabecd4` CI runs lint only.

The drift is already observable. Between `ff2a795` (2026-06-07) and `394fe81` (2026-08-26), upstream edited
pre-2025 history:

| Upstream change | Example |
|---|---|
| Home/away swapped | 1919 China – Philippines rows |
| Matches inserted | Three 2005 matches |
| Tournament relabelled | 2012 Taiwan – Guam: "Friendly" → "Philippine Peace Cup" |
| Team renamed (2026-06-06) | "China PR" → "China" |

| Metric | Pinned `ff2a795` | Upstream `394fe81` | Detected by repo? |
|---|--:|--:|:-:|
| Backtest skill | +0.0392 | +0.0389 | No |
| Stage ECE | 0.020 | 0.016 | No |

The effect is small today. But every Dixon-Coles parameter shifts silently, and an upstream rename would
drop a team from every training set without an error.

**Recommended fix (out of scope here).** Pin the upstream commit, verify sha256 on fetch, and add a
regression test on the snapshot, with tolerances set from the measured cross-platform spread (not ±1e-4).

<details>
<summary>Other gaps noted at <code>fabecd4</code></summary>

The test counts disagree: the README says 123 ([`README.md:134`][r134]), the CI comment says 115
([`ci.yml:17`][ci17]), and the suite has 144 `def test_` functions.

</details>

---

## 6. Audit artifact SHA

The artifact is **one fixed commit SHA**, stated in the submission (a file cannot contain its own hash). The
SHA is authoritative; a branch can move. The commit is also tagged `audit-fabecd4-v2`, which supersedes
`audit-fabecd4-v1`: same findings and numbers, earlier layout. To verify the anchoring:

```bash
git merge-base --is-ancestor fabecd4736461a216d974fff8cc8266e6b50f9f9 <audit-sha>              # exit 0
git diff --name-only fabecd4736461a216d974fff8cc8266e6b50f9f9 <audit-sha> | grep -v '^audit/'   # no output
```

Because the audit descends from `fabecd4`, it also keeps `fabecd4` reachable even if `main` is rewritten.

---

## Appendix: evidence index

| Path | Contents |
|---|---|
| [`outputs/results-pinned-ff2a795/`](outputs/results-pinned-ff2a795/) | Claims 1–3 on the pinned data (macOS) |
| [`outputs/results-upstream-394fe81/`](outputs/results-upstream-394fe81/) | The same claims on today's upstream data (the §5 drift) |
| [`outputs/club-odds-fetched-2026-10-03/`](outputs/club-odds-fetched-2026-10-03/) | Claims 4 and 6 |
| [`outputs/sbc-synthetic/`](outputs/sbc-synthetic/) | Claim 5, full rank histograms |
| [`outputs/bayesian-vs-mle/`](outputs/bayesian-vs-mle/) | Finding 3, with sampler diagnostics |
| [`outputs/cross-platform/`](outputs/cross-platform/) | Linux runs of `fabecd4` + platform comparison |
| [`reproduce.sh`](reproduce.sh) | The reproduction script verified in §1 |

[ref]: https://github.com/Ilan-07/wc2026/tree/fabecd4736461a216d974fff8cc8266e6b50f9f9
[readme-val]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/README.md#L39-L46
[r23]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/README.md#L23
[r45]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/README.md#L45
[r134]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/README.md#L134
[f18]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/FINDINGS.md#L18
[f42]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/FINDINGS.md#L42
[tb]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/src/wc2026/evaluate/tournament_backtest.py
[sr]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/src/wc2026/evaluate/stage_reliability.py
[l91]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/src/wc2026/data/loaders.py#L91
[dc190]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/src/wc2026/ratings/dixon_coles.py#L190
[p45]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/predict.py#L45
[c25]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/src/wc2026/config.py#L25
[b18]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/blend_weight_fit.py#L18
[s36]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/sbc_validate.py#L36
[sl]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/score_live.py
[t41]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/tests/test_lane3.py#L41
[ci17]: https://github.com/Ilan-07/wc2026/blob/fabecd4736461a216d974fff8cc8266e6b50f9f9/.github/workflows/ci.yml#L17
[harness]: https://github.com/Ilan-07/wc2026/blob/6f54e0616a30c1eea8507abbed045a97660d63b5/.github/workflows/audit-linux-fabecd4.yml
[run]: https://github.com/Ilan-07/wc2026/actions/runs/37105232750
