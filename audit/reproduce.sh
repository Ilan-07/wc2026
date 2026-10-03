#!/usr/bin/env bash
# Clean-machine reproduction of the README headline validation claims at fabecd4.
# Bounded: ~5 min on a laptop (+~3 min optional SBC), no API keys. Run from an empty directory.
set -euo pipefail

REF=fabecd4736461a216d974fff8cc8266e6b50f9f9   # the immutable reference — never the moving main branch
git clone https://github.com/Ilan-07/wc2026 && cd wc2026
git -c advice.detachedHead=false checkout --detach "$REF"
test "$(git rev-parse HEAD)" = "$REF" || { echo "not at $REF — refusing to run"; exit 1; }
test -z "$(git status --porcelain)"   || { echo "tree differs from $REF — refusing to run"; exit 1; }

python3.12 -m venv .venv && . .venv/bin/activate          # any Python >= 3.11 works with the lock
pip install numpy==2.4.2 scipy==1.17.0 pandas==2.3.3        # subset of requirements-lock.txt
pip install -e . --no-deps

# Pin the results feed to the upstream commit live when the claims were written (2026-06-07).
# `python fetch_data.py` would instead pull martj42 *master*, which has since changed (see AUDIT.md §5).
RJ=https://raw.githubusercontent.com/martj42/international_results/ff2a795faf5e2a94f3c72717db932d90368ea5bc
mkdir -p data/raw/odds
curl -sSL -o data/raw/results.csv   "$RJ/results.csv"
curl -sSL -o data/raw/shootouts.csv "$RJ/shootouts.csv"
echo "27d2d19b259d30b36f70a34f3f0e0f54ff3b7c90776de7c24659fb925c7d9992  data/raw/results.csv"   | shasum -a 256 -c
echo "cd66ba9cf02032ef2c5fa93717ade300f0e4161aa1a3d2c5dba5eee989acc9de  data/raw/shootouts.csv" | shasum -a 256 -c

# Club odds (football-data.co.uk has no versioned URLs; these are completed seasons).
for lg in E0 D1 SP1 I1 F1; do for s in 2122 2223 2324; do
  curl -sSL -o "data/raw/odds/${lg}_${s}.csv" "https://www.football-data.co.uk/mmz4281/${s}/${lg}.csv"
done; done

export PYTHONPATH=src
python cli.py backtest            # 9-tournament RPS 0.195 vs 0.234, skill +0.039, 8/9 positive   (~90 s)
python cli.py probe               # match-level ECE ~0.03 (the "overall ... ECE" line)             (~80 s)
python cli.py stage-reliability   # pooled Brier 0.104, ECE 0.020                                  (~55 s)
python cli.py validate            # model 0.204 vs market 0.190 RPS                                (~3 s)
python cli.py blend-weight        # CV model weight 0.00                                           (~3 s)

# Optional: simulation-based calibration of the Bayesian sampler (synthetic data, seed 0, ~3 min).
pip install pymc==6.0.1 arviz==1.1.0
python cli.py sbc                 # mu0 0.954, home 0.026 (flagged), sigma_att 0.173, att_0 0.637

# Optional: hierarchical Bayesian vs MLE on WC2022 (FINDINGS; ~20 s once PyMC is installed).
python bayesian_ablation.py       # Bayesian 0.2078 vs MLE 0.2241 (backtest-config MLE scores 0.2145)
