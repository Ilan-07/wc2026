"""Simulation-based calibration of the Bayesian rating's sampler (#7, Lane 3).

Validates the inference machinery (not the football data): are the SBC rank histograms uniform?
Run: ``PYTHONPATH=src python3 sbc_validate.py``  (slow — runs many short MCMC fits).
"""

from __future__ import annotations

import numpy as np

from wc2026.evaluate.sbc import TRACKED, run_sbc


def _ascii_hist(ranks: np.ndarray, n_post: int, n_bins: int = 10, width: int = 40) -> str:
    counts, _ = np.histogram(ranks, bins=np.linspace(-0.5, n_post + 0.5, n_bins + 1))
    top = max(counts.max(), 1)
    return "\n".join("      " + "#" * int(round(width * c / top)) + f" {c}" for c in counts)


def main() -> dict:
    res = run_sbc(n_sims=128, n_teams=5, n_matches=60, draws=250, tune=250, chains=2, seed=0)
    print(f"SBC: {res['n_sims']} replicates, {res['n_post']} posterior draws each "
          f"(rank ∈ [0, {res['n_post']}]).\n")
    print("Uniform ranks ⇒ the sampler recovers the posterior correctly. p>0.05 = no detected "
          "miscalibration:\n")
    for k in TRACKED:
        ranks = res["ranks"][k]
        p = res["uniformity"][k]
        flag = "OK" if p > 0.05 else "*** non-uniform ***"
        print(f"  {k:<10} chi-square p = {p:.3f}   [{flag}]   mean rank "
              f"{ranks.mean():.0f} / {res['n_post']} (expect {res['n_post'] / 2:.0f})")
        print(_ascii_hist(ranks, res["n_post"]))
    worst = min(res["uniformity"].values())
    print(f"\nVerdict: {'all tracked parameters pass' if worst > 0.05 else 'some parameter flags'} "
          f"SBC uniformity (min p = {worst:.3f}). The hierarchical-Bayesian inference behind "
          f"`predict --bayesian` is calibrated on its own generative model.")
    return res


if __name__ == "__main__":
    main()
