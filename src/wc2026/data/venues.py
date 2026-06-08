"""WC2026 venue facts — altitude, coordinates, climate (plan Phase B conditions).

This is the one "context" data layer that is genuinely *orthogonal* to match results and needs no
scraping or API key — it's public geography. The standout signal is **altitude**: three venues are
in Mexico, and Estadio Azteca (Mexico City) sits at ~2,240 m, where sea-level teams measurably tire.

Keyed by the ``city`` field as it appears in results.csv's WC2026 fixtures (note the Mexican metro
suburbs: Zapopan = Guadalajara/Akron, Guadalupe = Monterrey/BBVA).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Venue:
    city: str
    stadium: str
    country: str
    lat: float
    lon: float
    altitude_m: int
    climate: str  # rough June/July descriptor


VENUES: dict[str, Venue] = {
    "Atlanta": Venue("Atlanta", "Mercedes-Benz Stadium", "USA", 33.755, -84.401, 320, "humid"),
    "Foxborough": Venue("Foxborough", "Gillette Stadium", "USA", 42.091, -71.264, 28, "temperate"),
    "Arlington": Venue("Arlington", "AT&T Stadium", "USA", 32.747, -97.093, 188, "hot"),
    "Houston": Venue("Houston", "NRG Stadium", "USA", 29.685, -95.411, 15, "hot-humid"),
    "Kansas City": Venue("Kansas City", "Arrowhead Stadium", "USA", 39.049, -94.484, 277, "hot"),
    "Inglewood": Venue("Inglewood", "SoFi Stadium", "USA", 33.953, -118.339, 30, "mild"),
    "Miami Gardens": Venue("Miami Gardens", "Hard Rock Stadium", "USA", 25.958, -80.239, 3, "hot-humid"),
    "East Rutherford": Venue("East Rutherford", "MetLife Stadium", "USA", 40.814, -74.074, 5, "temperate"),
    "Philadelphia": Venue("Philadelphia", "Lincoln Financial Field", "USA", 39.901, -75.168, 12, "temperate"),
    "Santa Clara": Venue("Santa Clara", "Levi's Stadium", "USA", 37.403, -121.969, 9, "mild"),
    "Seattle": Venue("Seattle", "Lumen Field", "USA", 47.595, -122.332, 17, "mild"),
    "Toronto": Venue("Toronto", "BMO Field", "Canada", 43.633, -79.418, 76, "temperate"),
    "Vancouver": Venue("Vancouver", "BC Place", "Canada", 49.277, -123.112, 3, "mild"),
    "Mexico City": Venue("Mexico City", "Estadio Azteca", "Mexico", 19.303, -99.150, 2240, "altitude"),
    "Zapopan": Venue("Zapopan", "Estadio Akron", "Mexico", 20.681, -103.463, 1566, "altitude"),
    "Guadalupe": Venue("Guadalupe", "Estadio BBVA", "Mexico", 25.669, -100.244, 450, "hot"),
}

# Approximate altitude (m) of each national team's usual home base, for altitude *adaptation*.
# Only teams with notably high home altitude matter; everyone else defaults to ~sea level.
HOME_ALTITUDE: dict[str, int] = {
    "Mexico": 2240, "Ecuador": 2850, "Colombia": 2640, "South Africa": 1750,
    "Iran": 1200, "Saudi Arabia": 600, "Austria": 190, "Switzerland": 400,
}
DEFAULT_HOME_ALTITUDE = 100


def venue_for_city(city: str) -> Venue | None:
    return VENUES.get(city)


def home_altitude(team: str) -> int:
    return HOME_ALTITUDE.get(team, DEFAULT_HOME_ALTITUDE)
