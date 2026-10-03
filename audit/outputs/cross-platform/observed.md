# Headline numbers by platform — same code (fix/reproducibility test gate), same pinned snapshot

Source: `[repro]` lines printed by tests/test_reproducibility.py (`pytest -rP`).
GitHub Actions run 37101389853 (ubuntu-latest, Python 3.12, locked numpy/scipy/pandas), attempts 1–3,
plus the earlier run 37101044738 (failed on the stage-reliability assertion with the values of "Linux B").

| Platform | skill vs uniform | pooled RPS | per-tourn. mean | match ECE | stage Brier | stage ECE | ECE sf |
|---|---|---|---|---|---|---|---|
| macOS (3 local runs, identical) | 0.039197 | 0.195139 | 0.187946 | 0.03111 | 0.104410 | 0.01969 | 0.0391 |
| Linux A (run 37101389853 attempt 1) | 0.039187 | 0.195149 | 0.187960 | 0.03052 | 0.104316 | 0.01913 | 0.0236 |
| Linux B (attempts 2, 3; run 37101044738) | 0.039087 | 0.195249 | 0.188111 | 0.03205 | 0.103986 | 0.01719 | 0.0139 |

Model vs market is stable to 2e-6 on all three (model 0.20428, market 0.18970, CV weight 0.00).
Each platform is deterministic run-to-run; the variation is across CPU/BLAS builds.
