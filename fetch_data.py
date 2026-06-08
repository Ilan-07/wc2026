"""Reproducible data acquisition (gap #28) — every source the system needs, in one place.

    python fetch_data.py            # fetch everything (skips files already present)
    python fetch_data.py --force    # re-download all

Run once after cloning. Live odds need an Odds API key (see wc2026.collective.odds_api); skipped
here if absent. Respects each source's public/open access — no scraping of ToS-restricted sites.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.parse
from pathlib import Path

RAW = Path("data/raw")
ODDS = RAW / "odds"

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
# football-data.co.uk club odds (5 leagues x 3 seasons) for fusion validation
LEAGUES = ["E0", "D1", "SP1", "I1", "F1"]
SEASONS = ["2122", "2223", "2324"]


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
    print("Club odds (football-data.co.uk):")
    for lg in LEAGUES:
        for sea in SEASONS:
            fetch(ODDS / f"{lg}_{sea}.csv", f"https://www.football-data.co.uk/mmz4281/{sea}/{lg}.csv", force)
    print("\nDone. Live outright odds: run `python cli.py odds` with an Odds API key set.")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
