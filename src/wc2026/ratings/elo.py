"""World Football Elo ratings.

A robust, well-calibrated baseline rating that also serves as a prior anchor for the
Dixon-Coles attack/defense parameters (see plan Tier A). The implementation follows the
eloratings.net conventions:

    E_i = 1 / (1 + 10 ** ((R_j - R_i - H_i) / 400))          # expected result for i
    R_i' = R_i + K * G * (S_i - E_i)                          # update after the match

where
  * ``H_i``  is the home-advantage bonus added to the home side's rating (0 at neutral
    World Cup venues, except for host nations playing at home),
  * ``S_i``  is the actual result (1 win / 0.5 draw / 0 loss),
  * ``K``    is the match-importance weight (friendly < qualifier < major tournament),
  * ``G``    is the goal-difference multiplier that rewards decisive wins.

The module is deliberately dependency-light (pure ``math``) so it can run anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Match-importance weights (K). Larger => ratings react faster to that match type.
K_BY_IMPORTANCE: dict[str, float] = {
    "friendly": 10.0,
    "nations_league": 25.0,
    "qualifier": 25.0,
    "continental": 40.0,  # Euro / Copa America etc.
    "confederations": 40.0,
    "world_cup": 60.0,
}

DEFAULT_RATING = 1500.0
DEFAULT_HOME_ADVANTAGE = 65.0  # ~Elo points; set to 0.0 for neutral venues.


def expected_score(rating_i: float, rating_j: float, home_advantage: float = 0.0) -> float:
    """Expected result for team *i* (in [0, 1]); ``home_advantage`` favours team *i*."""
    return 1.0 / (1.0 + 10.0 ** ((rating_j - rating_i - home_advantage) / 400.0))


def goal_difference_multiplier(goals_i: int, goals_j: int) -> float:
    """eloratings.net goal-difference multiplier G.

    G = 1 for a one-goal margin, 1.5 for a two-goal margin, and
    (11 + margin) / 8 for margins of three or more.
    """
    margin = abs(goals_i - goals_j)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def _result_score(goals_i: int, goals_j: int) -> float:
    if goals_i > goals_j:
        return 1.0
    if goals_i < goals_j:
        return 0.0
    return 0.5


@dataclass
class EloModel:
    """Maintains a table of team Elo ratings and updates them match by match.

    Ratings persist across calls so you can stream a chronological match history through
    :meth:`update`. Use :meth:`win_probability` for a quick W/D/L-free favourite estimate.
    """

    k_by_importance: dict[str, float] = field(default_factory=lambda: dict(K_BY_IMPORTANCE))
    default_rating: float = DEFAULT_RATING
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, home: str, away: str, neutral: bool = True) -> float:
        h = 0.0 if neutral else self.home_advantage
        return expected_score(self.rating(home), self.rating(away), home_advantage=h)

    def win_probability(self, home: str, away: str, neutral: bool = True) -> float:
        """Probability the *home* team wins (draws split evenly). Convenience only;
        the Dixon-Coles model is the proper source of W/D/L probabilities."""
        return self.expected_score(home, away, neutral=neutral)

    def update(
        self,
        home: str,
        away: str,
        goals_home: int,
        goals_away: int,
        importance: str = "friendly",
        neutral: bool = True,
    ) -> tuple[float, float]:
        """Apply one match result and return the two teams' new ratings."""
        k = self.k_by_importance.get(importance, self.k_by_importance["friendly"])
        h = 0.0 if neutral else self.home_advantage

        r_home, r_away = self.rating(home), self.rating(away)
        e_home = expected_score(r_home, r_away, home_advantage=h)
        s_home = _result_score(goals_home, goals_away)
        g = goal_difference_multiplier(goals_home, goals_away)

        delta = k * g * (s_home - e_home)
        new_home = r_home + delta
        new_away = r_away - delta  # zero-sum update
        self.ratings[home] = new_home
        self.ratings[away] = new_away
        return new_home, new_away

    def fit(self, matches) -> EloModel:
        """Stream an iterable of match dicts (chronological order) through :meth:`update`.

        Each match needs: ``home_team``, ``away_team``, ``home_score``, ``away_score``;
        optional ``importance`` and ``neutral``.
        """
        for m in matches:
            self.update(
                m["home_team"],
                m["away_team"],
                int(m["home_score"]),
                int(m["away_score"]),
                importance=m.get("importance", "friendly"),
                neutral=bool(m.get("neutral", True)),
            )
        return self
