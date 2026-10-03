"""Pinned evaluation snapshot — the frozen data every published validation number is computed on.

The live forecast reads ``data/raw/results.csv``, which ``predict --refresh`` overwrites from the
``martj42/international_results`` *master* branch every tick. That feed is mutable: upstream has edited
historical rows (home/away swaps, inserted matches, relabelled tournaments, team renames), which
silently moves every backtest number. So validation runs on an immutable, hash-checked snapshot kept
apart from the live feed:

    data/snapshots/martj42-ff2a795/{results,shootouts}.csv   # upstream commit ff2a795 (2026-06-07)
    data/snapshots/club-odds/<LEAGUE>_<SEASON>.csv            # football-data.co.uk, hash-checked

``fetch_snapshot()`` downloads and verifies it; ``use_snapshot()`` points the loaders at it. The
validation subcommands in ``cli.py`` call ``use_snapshot()`` unless ``--live-data`` is passed.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from . import loaders

ROOT = Path(__file__).resolve().parents[3]

RESULTS_COMMIT = "ff2a795faf5e2a94f3c72717db932d90368ea5bc"  # last upstream commit before 2026-06-08
RESULTS_DIR = ROOT / "data" / "snapshots" / f"martj42-{RESULTS_COMMIT[:7]}"
RESULTS_SHA256 = {
    "results.csv": "27d2d19b259d30b36f70a34f3f0e0f54ff3b7c90776de7c24659fb925c7d9992",
    "shootouts.csv": "cd66ba9cf02032ef2c5fa93717ade300f0e4161aa1a3d2c5dba5eee989acc9de",
}

# football-data.co.uk has no versioned URLs; the seasons are complete, so the bytes are pinned by hash.
ODDS_DIR = ROOT / "data" / "snapshots" / "club-odds"
ODDS_SHA256 = {
    "D1_2122.csv": "034d0e1296e944dddf25e3c91b85ef5bd1ad25661b89d73878f5c73bfea4b9fc",
    "D1_2223.csv": "b716517b25033ce8e0243e967aa961876425a4c41e97c8aa8fa59f2aca061341",
    "D1_2324.csv": "1de8bc12a133dbbf71bc6ffda62082642f087f61b1b7c15ae9f2d320876fb07e",
    "E0_2122.csv": "335afcbabeb2939fa10ab39ba3e8215072d0b577cb8d0705c1e44c56e934e703",
    "E0_2223.csv": "8442792d3b614c94ea3cf381bd2736805889cc1713169035368fff19c3d02380",
    "E0_2324.csv": "b2e057b0ed959f198b0f63d2391c01239f3608e6de5db68edab3f88e04d07ff3",
    "F1_2122.csv": "9e9eaefb8704264f8d55a718a119d940de4af7d7b4217ac4097b957406e16213",
    "F1_2223.csv": "18ea07d922e8b6360b81189637efd5aaf44779765db856bff208d5b1acf69b2f",
    "F1_2324.csv": "ca0a4bf51d5b1ccfd0e1558ea41de6bb829df6c96c759888a87248d296d5adb7",
    "I1_2122.csv": "6a9555d5a90e652a883fd2862a52545c06e264016cd770347c067a6d7c993170",
    "I1_2223.csv": "9960837f37a97ae700f876c87b85a8b7a2f5a5d05a53077830c21b9db8880c45",
    "I1_2324.csv": "343a89db026fe4204ebeabd663cdfee8d02a45457828fc1cb63fa630ca170723",
    "SP1_2122.csv": "2d9e619700712823d15716c07715133f8c3b1a09046930dcfc785a38633dee2e",
    "SP1_2223.csv": "53348112526b5165539df240be350083989bb258f30c0ff7cacbc0165ca7e4b3",
    "SP1_2324.csv": "5323f596c7ed318e4bce95557a369d809d98501fa499ea5f41350cd45cf3c7d9",
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sources() -> list[tuple[Path, str, str]]:
    """(dest, url, expected sha256) for every snapshot file."""
    gh = f"https://raw.githubusercontent.com/martj42/international_results/{RESULTS_COMMIT}"
    out = [(RESULTS_DIR / f, f"{gh}/{f}", h) for f, h in RESULTS_SHA256.items()]
    for f, h in ODDS_SHA256.items():
        league, season = f.removesuffix(".csv").split("_")
        out.append((ODDS_DIR / f, f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv", h))
    return out


def verify() -> list[str]:
    """Problems with the local snapshot (missing files or hash mismatches); empty if it is intact."""
    problems = []
    for dest, _, want in _sources():
        if not dest.exists():
            problems.append(f"missing {dest.relative_to(ROOT)}")
        elif (got := _sha256(dest)) != want:
            problems.append(f"sha256 mismatch {dest.relative_to(ROOT)}: {got[:12]}… != {want[:12]}…")
    return problems


def fetch_snapshot(force: bool = False) -> None:
    """Download any missing snapshot file, then fail loudly if any hash does not match."""
    for dest, url, _ in _sources():
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  get  {dest.relative_to(ROOT)}")
        subprocess.run(["curl", "-sSfL", "--max-time", "60", "-o", str(dest), url], check=True)
    if problems := verify():
        raise RuntimeError("evaluation snapshot failed verification:\n  " + "\n  ".join(problems))


def use_snapshot(results: bool = True) -> None:
    """Verify the snapshot and point the results loader at it (``results=False`` keeps the live feed;
    club odds are always read from the snapshot). Raises if the snapshot is not intact."""
    if problems := verify():
        raise RuntimeError(
            "evaluation snapshot missing or modified — run `python fetch_data.py` first:\n  "
            + "\n  ".join(problems)
        )
    if results:
        loaders.DEFAULT_RESULTS = RESULTS_DIR / "results.csv"
