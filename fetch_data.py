"""Reproducible data acquisition (gap #28) — every source the system needs, in one place.

    PYTHONPATH=src python fetch_data.py            # fetch everything (skips files already present)
    PYTHONPATH=src python fetch_data.py --force    # re-download all

Run once after cloning. Two kinds of data land:

* ``data/snapshots/`` — the **pinned, sha256-verified evaluation snapshot** (results/shootouts at a
  fixed upstream commit + the club-odds seasons). Every validation number in README/FINDINGS is
  computed on this; the download fails loudly if a byte differs. See ``wc2026.data.snapshot``.
* ``data/raw/`` — the **live** feeds (results ``master``, squads, knockout structure) that the forecast
  refreshes; these move by design.

Live odds need an Odds API key (see wc2026.collective.odds_api); skipped here if absent. Respects each
source's public/open access — no scraping of ToS-restricted sites.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.parse
from pathlib import Path

RAW = Path("data/raw")

GH = "https://raw.githubusercontent.com"
WIKI = "https://en.wikipedia.org/w/api.php"

# (dest, url)
FILES = [
    (RAW / "results.csv", f"{GH}/martj42/international_results/master/results.csv"),
    (RAW / "shootouts.csv", f"{GH}/martj42/international_results/master/shootouts.csv"),
    (RAW / "sb_competitions.json", f"{GH}/statsbomb/open-data/master/data/competitions.json"),
]
# Wikipedia articles -> raw json
WIKI_PAGES = {
    "wc2026_squads.json": "2026 FIFA World Cup squads",
    "wc2018_squads.json": "2018 FIFA World Cup squads",
    "wc2022_squads.json": "2022 FIFA World Cup squads",
    "wc2026_knockout.json": "2026 FIFA World Cup knockout stage",
}


def fetch(dest: Path, url: str, force: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  skip {dest} (exists)")
        return
    print(f"  get  {dest}")
    subprocess.run(["curl", "-sSL", "--max-time", "60", "-o", str(dest), url], check=True)


def main(force: bool = False) -> None:
    print("Core datasets:")
    for dest, url in FILES:
        fetch(dest, url, force)
    print("Wikipedia (squads + knockout structure):")
    for fname, title in WIKI_PAGES.items():
        q = urllib.parse.quote(title)
        url = (f"{WIKI}?action=query&format=json&prop=revisions&rvprop=content"
               f"&rvslots=main&titles={q}&redirects=1")
        fetch(RAW / fname, url, force)
    print("Pinned evaluation snapshot (results @ fixed commit + club odds, sha256-verified):")
    from wc2026.data.snapshot import fetch_snapshot
    fetch_snapshot(force)
    print("\nDone. Live outright odds: run `python cli.py odds` with an Odds API key set.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
