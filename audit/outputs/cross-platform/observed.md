# Headline numbers by platform

## Primary evidence: unmodified `fabecd4` code, pinned sha256-verified data

| Platform | Source | skill vs uniform | pooled RPS | match ECE | stage Brier | stage ECE | ECE sf |
|---|---|---|---|---|---|---|---|
| macOS (local, 3 runs, identical) | `results-pinned-ff2a795/`, `reproduce.sh` ×2 | +0.0392 | 0.1951 | 0.031 | 0.1044 | 0.020 | 0.039 |
| Linux, AMD EPYC 7763 (3 runs, identical) | GitHub Actions run 37105232750 ([log](linux-fabecd4-run-37105232750.txt)) | +0.0392 | 0.1951 | 0.031 | 0.1043 | 0.019 | 0.024 |

Values are as printed by the `fabecd4` CLI (`cli.py backtest | probe | stage-reliability`). Each run checks
out `fabecd4`, asserts `git rev-parse HEAD` equals it with a clean tree, and logs the CPU model.
Model vs market is identical on both platforms (model 0.2043, market 0.1897).

## Corroborating only: NOT `fabecd4`'s tree

GitHub Actions runs 37101044738 and 37101389853 (attempts 1–3) ran the follow-up branch
`fix/reproducibility` (commits `d8a6d71`, `ec607aa`). On that branch the evaluation code
(`src/wc2026/{evaluate,ratings,model,simulate,fusion}`, `data/loaders.py`, `fusion_validate.py`) is
byte-identical to `fabecd4`, apart from the club-odds directory path in `collective/market.py`. It also
installs the full lock file, and the runner CPU was not recorded. A second Linux outcome appeared there,
and it was not seen in the three `fabecd4` runs above:

| Platform | skill vs uniform | match ECE | stage Brier | stage ECE | ECE sf |
|---|---|---|---|---|---|
| Linux "A" (attempt 1) | 0.039187 | 0.0305 | 0.10432 | 0.0191 | 0.024 |
| Linux "B" (attempts 2–3, run 37101044738) | 0.039087 | 0.0321 | 0.10399 | 0.0172 | 0.014 |

Linux "A" matches the `fabecd4` AMD EPYC 7763 run at printed precision. Linux "B" is reported as an
indication that a third floating-point outcome exists. It is not counted as a measurement of `fabecd4`.
