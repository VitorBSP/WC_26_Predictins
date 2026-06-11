from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



FEATURE_COLUMNS = [
    "elo_diff",
    "attack_diff",
    "defense_diff",
    "form_diff",
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_total_points",
    "away_total_points",
    "fifa_points_diff",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "neutral",
    "is_non_neutral",
    "tournament_weight",
    "home_attack",
    "away_attack",
    "home_defense",
    "away_defense",
    "home_form",
    "away_form",
    "home_decay",
    "away_decay",
    "home_matches_played",
    "away_matches_played",
]

DEFAULT_MARKET_PROBABILITIES = {
    "market_home_prob": 0.45,
    "market_draw_prob": 0.25,
    "market_away_prob": 0.30,
}

DEFAULT_RANK = 100.0
DEFAULT_TOTAL_POINTS = 1000.0
DEFAULT_ELO = 1500.0
DEFAULT_GOAL_STRENGTH = 1.20
DEFAULT_FORM = 0.50


@dataclass
class TeamState:
    elo: float = DEFAULT_ELO
    attack: float = DEFAULT_GOAL_STRENGTH
    defense: float = DEFAULT_GOAL_STRENGTH
    form: float = DEFAULT_FORM
    matches_played: int = 0
    last_match_date: pd.Timestamp | None = None


@dataclass
class MatchSimulation:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: str | None
    stage: str
    group: str | None


@dataclass
class TournamentSimulation:
    champion: str
    runner_up: str
    semifinalists: list[str]
    round_of_32_teams: list[str]
    group_advancers: list[str]
    playoff_qualifiers: list[str]


class RankingLookup:
    def __init__(self, rankings_data: pd.DataFrame | None) -> None:
        self.rankings_by_team: dict[str, pd.DataFrame] = {}

        if rankings_data is None or rankings_data.empty:
            return
            
        # Cria a variável 'rank' baseada no 'total_points' do maior para o menor

        if "total_points" in rankings_data.columns:
            rankings_data["rank"] = (
                rankings_data.groupby("date")["total_points"]
                .rank(method="min", ascending=False)
            )

        standardized_data = standardize_rankings_columns(rankings_data)
        
        

        for team, team_rankings in standardized_data.groupby("team"):
            ordered_rankings = team_rankings.sort_values("rank_date").reset_index(drop=True)
            self.rankings_by_team[team] = ordered_rankings

    def get(self, team: str, match_date: pd.Timestamp) -> tuple[float, float]:
        team_key = normalize_team_name(team)
        team_rankings = self.rankings_by_team.get(team_key)

        if team_rankings is None or team_rankings.empty:
            return DEFAULT_RANK, DEFAULT_TOTAL_POINTS

        dates = team_rankings["rank_date"].to_numpy(dtype="datetime64[ns]")
        position = np.searchsorted(dates, np.datetime64(match_date), side="right") - 1

        if position < 0:
            return DEFAULT_RANK, DEFAULT_TOTAL_POINTS

        ranking_row = team_rankings.iloc[position]
        return float(ranking_row["rank"]), float(ranking_row["total_points"])


class OddsLookup:
    def __init__(self, odds_data: pd.DataFrame | None) -> None:
        self.market_by_key: dict[tuple[str, str, str], dict[str, float]] = {}

        if odds_data is None or odds_data.empty:
            return

        standardized_data = standardize_odds_columns(odds_data)
        for _, row in standardized_data.iterrows():
            market_probabilities = odds_to_market_probabilities(
                row["home_win_odds"],
                row["draw_odds"],
                row["away_win_odds"],
            )
            match_key = (
                str(pd.Timestamp(row["match_date"]).date()),
                normalize_team_name(row["home_team"]),
                normalize_team_name(row["away_team"]),
            )
            self.market_by_key[match_key] = market_probabilities

    def get(self, match_date: pd.Timestamp, home_team: str, away_team: str) -> dict[str, float]:
        match_key = (
            str(pd.Timestamp(match_date).date()),
            normalize_team_name(home_team),
            normalize_team_name(away_team),
        )

        if match_key in self.market_by_key:
            return self.market_by_key[match_key]

        reverse_key = (
            str(pd.Timestamp(match_date).date()),
            normalize_team_name(away_team),
            normalize_team_name(home_team),
        )

        if reverse_key in self.market_by_key:
            reverse_market = self.market_by_key[reverse_key]
            return {
                "market_home_prob": reverse_market["market_away_prob"],
                "market_draw_prob": reverse_market["market_draw_prob"],
                "market_away_prob": reverse_market["market_home_prob"],
            }

        return DEFAULT_MARKET_PROBABILITIES.copy()


def normalize_team_name(team: Any) -> str:
    if pd.isna(team):
        return ""
    return str(team).strip()


def read_csv(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_csv(file_path)


def standardize_matches_columns(matches_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = matches_data.copy()
    rename_map = {
        "date": "match_date",
        "home_score": "home_goals",
        "away_score": "away_goals",
    }
    standardized_data = standardized_data.rename(columns=rename_map)

    required_columns = [
        "match_date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "tournament",
        "neutral",
    ]
    missing_columns = [column for column in required_columns if column not in standardized_data.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no histórico: {missing_columns}")

    standardized_data["match_date"] = pd.to_datetime(standardized_data["match_date"])
    standardized_data["home_team"] = standardized_data["home_team"].map(normalize_team_name)
    standardized_data["away_team"] = standardized_data["away_team"].map(normalize_team_name)
    standardized_data["home_goals"] = standardized_data["home_goals"].astype(int)
    standardized_data["away_goals"] = standardized_data["away_goals"].astype(int)
    standardized_data["neutral"] = standardized_data["neutral"].map(parse_bool).astype(int)
    standardized_data["tournament"] = standardized_data["tournament"].fillna("Unknown").astype(str)

    return standardized_data.sort_values("match_date").reset_index(drop=True)


def standardize_rankings_columns(rankings_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = rankings_data.copy()
    rename_map = {
        "date": "rank_date",
        "country_full": "team",
        "country": "team",
        "rank_points": "total_points",
    }
    standardized_data = standardized_data.rename(columns=rename_map)

    required_columns = ["rank_date", "team", "rank", "total_points"]
    missing_columns = [column for column in required_columns if column not in standardized_data.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no ranking FIFA: {missing_columns}")

    standardized_data["rank_date"] = pd.to_datetime(standardized_data["rank_date"])
    standardized_data["team"] = standardized_data["team"].map(normalize_team_name)
    standardized_data["rank"] = pd.to_numeric(standardized_data["rank"], errors="coerce").fillna(DEFAULT_RANK)
    standardized_data["total_points"] = pd.to_numeric(
        standardized_data["total_points"],
        errors="coerce",
    ).fillna(DEFAULT_TOTAL_POINTS)

    return standardized_data.sort_values(["team", "rank_date"]).reset_index(drop=True)


def standardize_odds_columns(odds_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = odds_data.copy()
    rename_map = {"date": "match_date"}
    standardized_data = standardized_data.rename(columns=rename_map)

    required_columns = [
        "match_date",
        "home_team",
        "away_team",
        "home_win_odds",
        "draw_odds",
        "away_win_odds",
    ]
    missing_columns = [column for column in required_columns if column not in standardized_data.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no arquivo de odds: {missing_columns}")

    standardized_data["match_date"] = pd.to_datetime(standardized_data["match_date"])
    standardized_data["home_team"] = standardized_data["home_team"].map(normalize_team_name)
    standardized_data["away_team"] = standardized_data["away_team"].map(normalize_team_name)

    return standardized_data


def standardize_schedule_columns(schedule_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = schedule_data.copy()
    rename_map = {"date": "match_date"}
    standardized_data = standardized_data.rename(columns=rename_map)

    required_columns = [
        "match_date",
        "home_team",
        "away_team",
        "stage",
        "tournament",
        "neutral",
    ]
    missing_columns = [column for column in required_columns if column not in standardized_data.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes na tabela da Copa: {missing_columns}")

    if "group" not in standardized_data.columns:
        standardized_data["group"] = None

    standardized_data["match_date"] = pd.to_datetime(standardized_data["match_date"])
    standardized_data["home_team"] = standardized_data["home_team"].map(normalize_team_name)
    standardized_data["away_team"] = standardized_data["away_team"].map(normalize_team_name)
    standardized_data["stage"] = standardized_data["stage"].fillna("").astype(str)
    standardized_data["group"] = standardized_data["group"].where(standardized_data["group"].notna(), None)
    standardized_data["tournament"] = standardized_data["tournament"].fillna("FIFA World Cup").astype(str)
    standardized_data["neutral"] = standardized_data["neutral"].map(parse_bool).astype(int)

    return standardized_data.sort_values("match_date").reset_index(drop=True)


def standardize_playoff_columns(playoff_data: pd.DataFrame) -> pd.DataFrame:
    standardized_data = playoff_data.copy()
    rename_map = {"date": "match_date"}
    standardized_data = standardized_data.rename(columns=rename_map)

    required_columns = [
        "match_order",
        "match_date",
        "home_team",
        "away_team",
        "winner_slot",
    ]
    missing_columns = [column for column in required_columns if column not in standardized_data.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no playoff_schedule: {missing_columns}")

    if "stage" not in standardized_data.columns:
        standardized_data["stage"] = "playoff"
    if "group" not in standardized_data.columns:
        standardized_data["group"] = None
    if "tournament" not in standardized_data.columns:
        standardized_data["tournament"] = "FIFA World Cup Play-Off Tournament"
    if "neutral" not in standardized_data.columns:
        standardized_data["neutral"] = 1

    standardized_data["match_order"] = standardized_data["match_order"].astype(int)
    standardized_data["match_date"] = pd.to_datetime(standardized_data["match_date"])
    standardized_data["home_team"] = standardized_data["home_team"].map(normalize_team_name)
    standardized_data["away_team"] = standardized_data["away_team"].map(normalize_team_name)
    standardized_data["winner_slot"] = standardized_data["winner_slot"].map(normalize_team_name)
    standardized_data["neutral"] = standardized_data["neutral"].map(parse_bool).astype(int)

    return standardized_data.sort_values(["match_order", "match_date"]).reset_index(drop=True)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    if isinstance(value, (int, float)):
        return bool(value)

    normalized_value = str(value).strip().lower()
    return normalized_value in {"true", "t", "1", "yes", "y", "sim", "s"}


def odds_to_market_probabilities(
    home_win_odds: float,
    draw_odds: float,
    away_win_odds: float,
) -> dict[str, float]:
    odds = np.array([home_win_odds, draw_odds, away_win_odds], dtype=float)
    if np.any(odds <= 1.0):
        return DEFAULT_MARKET_PROBABILITIES.copy()

    implied_probabilities = 1.0 / odds
    normalized_probabilities = implied_probabilities / implied_probabilities.sum()

    return {
        "market_home_prob": float(normalized_probabilities[0]),
        "market_draw_prob": float(normalized_probabilities[1]),
        "market_away_prob": float(normalized_probabilities[2]),
    }


def get_tournament_weight(tournament: str) -> float:
    tournament_name = tournament.lower()

    if "world cup" in tournament_name and "qualification" not in tournament_name:
        return 1.80
    if "copa américa" in tournament_name or "copa america" in tournament_name:
        return 1.60
    if "euro" in tournament_name:
        return 1.60
    if "africa cup" in tournament_name or "afcon" in tournament_name:
        return 1.55
    if "asian cup" in tournament_name:
        return 1.50
    if "qualification" in tournament_name or "qualifier" in tournament_name:
        return 1.30
    if "nations league" in tournament_name:
        return 1.15
    if "friendly" in tournament_name:
        return 0.85

    return 1.00


def get_or_create_state(states: dict[str, TeamState], team: str) -> TeamState:
    team_key = normalize_team_name(team)
    if team_key not in states:
        states[team_key] = TeamState()
    return states[team_key]


def calculate_decay(last_match_date: pd.Timestamp | None, match_date: pd.Timestamp) -> float:
    if last_match_date is None:
        return 0.0

    days_since_last_match = max((match_date - last_match_date).days, 0)
    return float(0.5 ** (days_since_last_match / 365.0))


def build_feature_row(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    neutral: int,
    tournament: str,
    states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    market_probabilities: dict[str, float],
) -> dict[str, float]:
    home_state = get_or_create_state(states, home_team)
    away_state = get_or_create_state(states, away_team)

    home_rank, home_total_points = ranking_lookup.get(home_team, match_date)
    away_rank, away_total_points = ranking_lookup.get(away_team, match_date)

    return {
        "elo_diff": home_state.elo - away_state.elo,
        "attack_diff": home_state.attack - away_state.attack,
        "defense_diff": away_state.defense - home_state.defense,
        "form_diff": home_state.form - away_state.form,
        "home_rank": home_rank,
        "away_rank": away_rank,
        "rank_diff": away_rank - home_rank,
        "home_total_points": home_total_points,
        "away_total_points": away_total_points,
        "fifa_points_diff": home_total_points - away_total_points,
        "market_home_prob": market_probabilities["market_home_prob"],
        "market_draw_prob": market_probabilities["market_draw_prob"],
        "market_away_prob": market_probabilities["market_away_prob"],
        "neutral": float(neutral),
        "is_non_neutral": float(1 - neutral),
        "tournament_weight": get_tournament_weight(tournament),
        "home_attack": home_state.attack,
        "away_attack": away_state.attack,
        "home_defense": home_state.defense,
        "away_defense": away_state.defense,
        "home_form": home_state.form,
        "away_form": away_state.form,
        "home_decay": calculate_decay(home_state.last_match_date, match_date),
        "away_decay": calculate_decay(away_state.last_match_date, match_date),
        "home_matches_played": float(home_state.matches_played),
        "away_matches_played": float(away_state.matches_played),
    }


def calculate_match_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def update_team_states(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    match_date: pd.Timestamp,
    tournament: str,
    states: dict[str, TeamState],
    home_result_override: float | None = None,
) -> None:
    home_state = get_or_create_state(states, home_team)
    away_state = get_or_create_state(states, away_team)

    expected_home_result = 1.0 / (1.0 + 10.0 ** (-(home_state.elo - away_state.elo) / 400.0))
    if home_result_override is None:
        home_result = actual_result_from_score(home_goals, away_goals)
    else:
        home_result = home_result_override

    tournament_weight = get_tournament_weight(tournament)
    goal_margin = abs(home_goals - away_goals)
    margin_multiplier = 1.0 if goal_margin == 0 else math.log(goal_margin + 1.0) + 1.0
    k_factor = 20.0 * tournament_weight
    elo_delta = k_factor * margin_multiplier * (home_result - expected_home_result)

    home_state.elo += elo_delta
    away_state.elo -= elo_delta

    alpha = min(0.08 * tournament_weight, 0.30)
    home_opponent_factor = max(0.60, away_state.elo / DEFAULT_ELO)
    away_opponent_factor = max(0.60, home_state.elo / DEFAULT_ELO)

    home_attack_signal = home_goals * home_opponent_factor
    away_attack_signal = away_goals * away_opponent_factor
    home_defense_signal = away_goals / home_opponent_factor
    away_defense_signal = home_goals / away_opponent_factor

    home_state.attack = exponential_update(home_state.attack, home_attack_signal, alpha)
    away_state.attack = exponential_update(away_state.attack, away_attack_signal, alpha)
    home_state.defense = exponential_update(home_state.defense, home_defense_signal, alpha)
    away_state.defense = exponential_update(away_state.defense, away_defense_signal, alpha)

    home_points = calculate_match_points(home_goals, away_goals) / 3.0
    away_points = calculate_match_points(away_goals, home_goals) / 3.0

    home_state.form = exponential_update(home_state.form, home_points, alpha)
    away_state.form = exponential_update(away_state.form, away_points, alpha)

    home_state.matches_played += 1
    away_state.matches_played += 1
    home_state.last_match_date = match_date
    away_state.last_match_date = match_date


def actual_result_from_score(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0


def exponential_update(previous_value: float, observed_value: float, alpha: float) -> float:
    return float((1.0 - alpha) * previous_value + alpha * observed_value)


def build_training_data(
    historical_matches: pd.DataFrame,
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, TeamState]]:
    states: dict[str, TeamState] = {}
    feature_rows: list[dict[str, float]] = []
    home_targets: list[int] = []
    away_targets: list[int] = []

    for _, match_row in historical_matches.iterrows():
        match_date = pd.Timestamp(match_row["match_date"])
        home_team = match_row["home_team"]
        away_team = match_row["away_team"]

        market_probabilities = odds_lookup.get(match_date, home_team, away_team)
        feature_row = build_feature_row(
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            neutral=int(match_row["neutral"]),
            tournament=match_row["tournament"],
            states=states,
            ranking_lookup=ranking_lookup,
            market_probabilities=market_probabilities,
        )

        feature_rows.append(feature_row)
        home_targets.append(int(match_row["home_goals"]))
        away_targets.append(int(match_row["away_goals"]))

        update_team_states(
            home_team=home_team,
            away_team=away_team,
            home_goals=int(match_row["home_goals"]),
            away_goals=int(match_row["away_goals"]),
            match_date=match_date,
            tournament=match_row["tournament"],
            states=states,
        )

    features_data = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    home_goals = pd.Series(home_targets, name="home_goals")
    away_goals = pd.Series(away_targets, name="away_goals")

    return features_data, home_goals, away_goals, states


def train_poisson_model(features_data: pd.DataFrame, goals: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "poisson",
                PoissonRegressor(
                    alpha=0.001,
                    max_iter=1000,
                ),
            ),
        ],
    )
    model.fit(features_data[FEATURE_COLUMNS], goals)
    return model


def predict_expected_goals(
    home_model: Pipeline,
    away_model: Pipeline,
    feature_row: dict[str, float],
) -> tuple[float, float]:
    features_data = pd.DataFrame([feature_row])[FEATURE_COLUMNS]
    home_lambda = float(home_model.predict(features_data)[0])
    away_lambda = float(away_model.predict(features_data)[0])

    return max(home_lambda, 0.05), max(away_lambda, 0.05)


def simulate_score(
    home_lambda: float,
    away_lambda: float,
    random_generator: Generator,
) -> tuple[int, int]:
    home_goals = int(random_generator.poisson(home_lambda))
    away_goals = int(random_generator.poisson(away_lambda))
    return min(home_goals, 10), min(away_goals, 10)


def simulate_match(
    home_team: str,
    away_team: str,
    match_date: pd.Timestamp,
    neutral: int,
    tournament: str,
    stage: str,
    group: str | None,
    states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    random_generator: Generator,
    knockout: bool = False,
) -> MatchSimulation:
    market_probabilities = odds_lookup.get(match_date, home_team, away_team)
    feature_row = build_feature_row(
        home_team=home_team,
        away_team=away_team,
        match_date=match_date,
        neutral=neutral,
        tournament=tournament,
        states=states,
        ranking_lookup=ranking_lookup,
        market_probabilities=market_probabilities,
    )
    home_lambda, away_lambda = predict_expected_goals(home_model, away_model, feature_row)
    home_goals, away_goals = simulate_score(home_lambda, away_lambda, random_generator)

    winner = infer_winner_from_score(home_team, away_team, home_goals, away_goals)
    home_result_override = None

    if knockout and winner is None:
        home_advancement_probability = home_lambda / (home_lambda + away_lambda)
        home_advancement_probability = float(np.clip(home_advancement_probability, 0.35, 0.65))
        winner = home_team if random_generator.random() < home_advancement_probability else away_team
        home_result_override = 0.55 if winner == home_team else 0.45

    update_team_states(
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        match_date=match_date,
        tournament=tournament,
        states=states,
        home_result_override=home_result_override,
    )

    return MatchSimulation(
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        winner=winner,
        stage=stage,
        group=group,
    )


def infer_winner_from_score(
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> str | None:
    if home_goals > away_goals:
        return home_team
    if away_goals > home_goals:
        return away_team
    return None


def simulate_playoffs(
    playoff_data: pd.DataFrame | None,
    states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    random_generator: Generator,
) -> tuple[dict[str, str], list[str]]:
    if playoff_data is None or playoff_data.empty:
        return {}, []

    standardized_data = standardize_playoff_columns(playoff_data)
    slots: dict[str, str] = {}
    playoff_qualifiers: list[str] = []

    for _, match_row in standardized_data.iterrows():
        home_team = resolve_slot(match_row["home_team"], slots)
        away_team = resolve_slot(match_row["away_team"], slots)

        match = simulate_match(
            home_team=home_team,
            away_team=away_team,
            match_date=pd.Timestamp(match_row["match_date"]),
            neutral=int(match_row["neutral"]),
            tournament=match_row["tournament"],
            stage=match_row["stage"],
            group=None,
            states=states,
            ranking_lookup=ranking_lookup,
            odds_lookup=odds_lookup,
            home_model=home_model,
            away_model=away_model,
            random_generator=random_generator,
            knockout=True,
        )

        if match.winner is None:
            raise RuntimeError("Jogo eliminatório terminou sem vencedor.")

        winner_slot = normalize_team_name(match_row["winner_slot"])
        slots[winner_slot] = match.winner

        if "is_final" in match_row and parse_bool(match_row["is_final"]):
            playoff_qualifiers.append(match.winner)

    return slots, playoff_qualifiers


def resolve_slot(value: str, slots: dict[str, str]) -> str:
    value_key = normalize_team_name(value)
    return slots.get(value_key, value_key)


def resolve_schedule_slots(schedule_data: pd.DataFrame, slots: dict[str, str]) -> pd.DataFrame:
    resolved_data = schedule_data.copy()
    resolved_data["home_team"] = resolved_data["home_team"].map(lambda value: resolve_slot(value, slots))
    resolved_data["away_team"] = resolved_data["away_team"].map(lambda value: resolve_slot(value, slots))
    return resolved_data


def simulate_group_stage(
    group_matches: pd.DataFrame,
    states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    random_generator: Generator,
) -> tuple[dict[str, list[dict[str, Any]]], list[str], list[str]]:
    group_tables: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    group_advancers: list[str] = []

    for _, match_row in group_matches.iterrows():
        group_name = str(match_row["group"])
        home_team = match_row["home_team"]
        away_team = match_row["away_team"]

        ensure_group_entry(group_tables, group_name, home_team)
        ensure_group_entry(group_tables, group_name, away_team)

        match = simulate_match(
            home_team=home_team,
            away_team=away_team,
            match_date=pd.Timestamp(match_row["match_date"]),
            neutral=int(match_row["neutral"]),
            tournament=match_row["tournament"],
            stage=match_row["stage"],
            group=group_name,
            states=states,
            ranking_lookup=ranking_lookup,
            odds_lookup=odds_lookup,
            home_model=home_model,
            away_model=away_model,
            random_generator=random_generator,
            knockout=False,
        )

        update_group_table(
            group_tables[group_name],
            match.home_team,
            match.away_team,
            match.home_goals,
            match.away_goals,
        )

    ranked_groups = {
        group_name: rank_group_table(table_entries, random_generator)
        for group_name, table_entries in group_tables.items()
    }

    third_places: list[dict[str, Any]] = []
    for group_name, ranked_table in ranked_groups.items():
        if len(ranked_table) < 4:
            continue

        group_advancers.extend([ranked_table[0]["team"], ranked_table[1]["team"]])
        third_entry = ranked_table[2].copy()
        third_entry["group"] = group_name
        third_places.append(third_entry)

    best_third_places = rank_third_places(third_places, random_generator)[:8]
    group_advancers.extend([entry["team"] for entry in best_third_places])

    group_winners = [ranked_table[0]["team"] for _, ranked_table in sorted(ranked_groups.items())]
    group_runners_up = [ranked_table[1]["team"] for _, ranked_table in sorted(ranked_groups.items())]
    best_third_teams = [entry["team"] for entry in best_third_places]
    round_of_32_teams = group_winners + group_runners_up + best_third_teams

    return ranked_groups, group_advancers, round_of_32_teams


def ensure_group_entry(
    group_tables: dict[str, dict[str, dict[str, Any]]],
    group_name: str,
    team: str,
) -> None:
    if team in group_tables[group_name]:
        return

    group_tables[group_name][team] = {
        "team": team,
        "played": 0,
        "points": 0,
        "wins": 0,
        "goals_for": 0,
        "goals_against": 0,
        "goal_difference": 0,
    }


def update_group_table(
    table_entries: dict[str, dict[str, Any]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    home_entry = table_entries[home_team]
    away_entry = table_entries[away_team]

    home_entry["played"] += 1
    away_entry["played"] += 1

    home_entry["goals_for"] += home_goals
    home_entry["goals_against"] += away_goals
    away_entry["goals_for"] += away_goals
    away_entry["goals_against"] += home_goals

    home_entry["goal_difference"] = home_entry["goals_for"] - home_entry["goals_against"]
    away_entry["goal_difference"] = away_entry["goals_for"] - away_entry["goals_against"]

    home_points = calculate_match_points(home_goals, away_goals)
    away_points = calculate_match_points(away_goals, home_goals)
    home_entry["points"] += home_points
    away_entry["points"] += away_points

    if home_points == 3:
        home_entry["wins"] += 1
    if away_points == 3:
        away_entry["wins"] += 1


def rank_group_table(
    table_entries: dict[str, dict[str, Any]],
    random_generator: Generator,
) -> list[dict[str, Any]]:
    ranked_entries = []
    for entry in table_entries.values():
        ranked_entry = entry.copy()
        ranked_entry["tie_breaker_noise"] = random_generator.random()
        ranked_entries.append(ranked_entry)

    return sorted(
        ranked_entries,
        key=lambda entry: (
            entry["points"],
            entry["goal_difference"],
            entry["goals_for"],
            entry["wins"],
            entry["tie_breaker_noise"],
        ),
        reverse=True,
    )


def rank_third_places(
    third_places: list[dict[str, Any]],
    random_generator: Generator,
) -> list[dict[str, Any]]:
    ranked_entries = []
    for entry in third_places:
        ranked_entry = entry.copy()
        ranked_entry["tie_breaker_noise"] = random_generator.random()
        ranked_entries.append(ranked_entry)

    return sorted(
        ranked_entries,
        key=lambda entry: (
            entry["points"],
            entry["goal_difference"],
            entry["goals_for"],
            entry["wins"],
            entry["tie_breaker_noise"],
        ),
        reverse=True,
    )


def create_generic_round_pairs(teams: list[str]) -> list[tuple[str, str]]:
    ordered_teams = list(teams)
    if len(ordered_teams) != 32:
        raise ValueError(f"Era esperado receber 32 classificados, mas foram recebidos {len(ordered_teams)}.")

    return [
        (ordered_teams[index], ordered_teams[-index - 1])
        for index in range(16)
    ]


def simulate_knockout_bracket(
    round_of_32_teams: list[str],
    start_date: pd.Timestamp,
    states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    random_generator: Generator,
) -> tuple[str, str, list[str]]:
    current_round = create_generic_round_pairs(round_of_32_teams)
    round_names = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"]
    semifinalists: list[str] = []
    runner_up = ""

    for round_index, round_name in enumerate(round_names):
        winners: list[str] = []

        if round_name == "semifinal":
            semifinalists = [team for pair in current_round for team in pair]

        for match_index, (home_team, away_team) in enumerate(current_round):
            match_date = start_date + pd.Timedelta(days=round_index * 4 + match_index % 4)
            match = simulate_match(
                home_team=home_team,
                away_team=away_team,
                match_date=match_date,
                neutral=1,
                tournament="FIFA World Cup",
                stage=round_name,
                group=None,
                states=states,
                ranking_lookup=ranking_lookup,
                odds_lookup=odds_lookup,
                home_model=home_model,
                away_model=away_model,
                random_generator=random_generator,
                knockout=True,
            )

            if match.winner is None:
                raise RuntimeError("Jogo de mata-mata terminou sem vencedor.")

            winners.append(match.winner)

            if round_name == "final":
                runner_up = away_team if match.winner == home_team else home_team

        if len(winners) == 1:
            return winners[0], runner_up, semifinalists

        current_round = [
            (winners[index], winners[index + 1])
            for index in range(0, len(winners), 2)
        ]

    raise RuntimeError("Não foi possível finalizar o mata-mata.")


def run_one_simulation(
    schedule_data: pd.DataFrame,
    playoff_data: pd.DataFrame | None,
    initial_states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    random_generator: Generator,
) -> TournamentSimulation:
    states = copy.deepcopy(initial_states)

    slots, playoff_qualifiers = simulate_playoffs(
        playoff_data=playoff_data,
        states=states,
        ranking_lookup=ranking_lookup,
        odds_lookup=odds_lookup,
        home_model=home_model,
        away_model=away_model,
        random_generator=random_generator,
    )

    resolved_schedule = resolve_schedule_slots(schedule_data, slots)
    group_matches = resolved_schedule[
        resolved_schedule["stage"].str.lower().str.contains("group")
    ].copy()

    if group_matches.empty:
        raise ValueError("A tabela da Copa não tem jogos de fase de grupos.")

    _, group_advancers, round_of_32_teams = simulate_group_stage(
        group_matches=group_matches,
        states=states,
        ranking_lookup=ranking_lookup,
        odds_lookup=odds_lookup,
        home_model=home_model,
        away_model=away_model,
        random_generator=random_generator,
    )

    knockout_start_date = group_matches["match_date"].max() + pd.Timedelta(days=1)
    champion, runner_up, semifinalists = simulate_knockout_bracket(
        round_of_32_teams=round_of_32_teams,
        start_date=knockout_start_date,
        states=states,
        ranking_lookup=ranking_lookup,
        odds_lookup=odds_lookup,
        home_model=home_model,
        away_model=away_model,
        random_generator=random_generator,
    )

    return TournamentSimulation(
        champion=champion,
        runner_up=runner_up,
        semifinalists=semifinalists,
        round_of_32_teams=round_of_32_teams,
        group_advancers=group_advancers,
        playoff_qualifiers=playoff_qualifiers,
    )


def run_simulations(
    schedule_data: pd.DataFrame,
    playoff_data: pd.DataFrame | None,
    initial_states: dict[str, TeamState],
    ranking_lookup: RankingLookup,
    odds_lookup: OddsLookup,
    home_model: Pipeline,
    away_model: Pipeline,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    random_generator = default_rng(seed)

    champion_counter: Counter[str] = Counter()
    runner_up_counter: Counter[str] = Counter()
    semifinal_counter: Counter[str] = Counter()
    round_of_32_counter: Counter[str] = Counter()
    group_advancement_counter: Counter[str] = Counter()
    playoff_qualification_counter: Counter[str] = Counter()
    champion_given_playoff_counter: Counter[str] = Counter()

    for _ in range(simulations):
        simulation = run_one_simulation(
            schedule_data=schedule_data,
            playoff_data=playoff_data,
            initial_states=initial_states,
            ranking_lookup=ranking_lookup,
            odds_lookup=odds_lookup,
            home_model=home_model,
            away_model=away_model,
            random_generator=random_generator,
        )

        champion_counter[simulation.champion] += 1
        runner_up_counter[simulation.runner_up] += 1

        for team in simulation.semifinalists:
            semifinal_counter[team] += 1
        for team in simulation.round_of_32_teams:
            round_of_32_counter[team] += 1
        for team in simulation.group_advancers:
            group_advancement_counter[team] += 1
        for team in simulation.playoff_qualifiers:
            playoff_qualification_counter[team] += 1

        if simulation.champion in simulation.playoff_qualifiers:
            champion_given_playoff_counter[simulation.champion] += 1

    return {
        "champion": champion_counter,
        "runner_up": runner_up_counter,
        "semifinal": semifinal_counter,
        "round_of_32": round_of_32_counter,
        "group_advancement": group_advancement_counter,
        "playoff_qualification": playoff_qualification_counter,
        "champion_given_playoff": champion_given_playoff_counter,
        "simulations": simulations,
    }


def save_counter_probabilities(
    counter: Counter[str],
    simulations: int,
    output_path: Path,
    probability_column: str,
) -> None:
    probability_rows = [
        {
            "team": team,
            "count": count,
            probability_column: count / simulations,
        }
        for team, count in counter.most_common()
    ]
    probabilities_data = pd.DataFrame(probability_rows)
    probabilities_data.to_csv(output_path, index=False)


def save_conditional_playoff_champion_probabilities(
    champion_given_playoff_counter: Counter[str],
    playoff_qualification_counter: Counter[str],
    output_path: Path,
) -> None:
    rows = []
    for team, qualification_count in playoff_qualification_counter.items():
        champion_count = champion_given_playoff_counter.get(team, 0)
        rows.append(
            {
                "team": team,
                "playoff_qualification_count": qualification_count,
                "champion_count": champion_count,
                "champion_probability_given_qualification": (
                    champion_count / qualification_count
                    if qualification_count > 0
                    else 0.0
                ),
            }
        )

    probabilities_data = pd.DataFrame(rows).sort_values(
        "champion_probability_given_qualification",
        ascending=False,
    )
    probabilities_data.to_csv(output_path, index=False)


def save_outputs(results: dict[str, Any], output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    simulations = int(results["simulations"])

    save_counter_probabilities(
        results["champion"],
        simulations,
        output_dir / "champion_probabilities.csv",
        "champion_probability",
    )
    save_counter_probabilities(
        results["runner_up"],
        simulations,
        output_dir / "runner_up_probabilities.csv",
        "runner_up_probability",
    )
    save_counter_probabilities(
        results["semifinal"],
        simulations,
        output_dir / "semifinal_probabilities.csv",
        "semifinal_probability",
    )
    save_counter_probabilities(
        results["round_of_32"],
        simulations,
        output_dir / "round_of_32_probabilities.csv",
        "round_of_32_probability",
    )
    save_counter_probabilities(
        results["group_advancement"],
        simulations,
        output_dir / "group_advancement_probabilities.csv",
        "group_advancement_probability",
    )
    save_counter_probabilities(
        results["playoff_qualification"],
        simulations,
        output_dir / "playoff_qualification_probabilities.csv",
        "playoff_qualification_probability",
    )
    save_conditional_playoff_champion_probabilities(
        results["champion_given_playoff"],
        results["playoff_qualification"],
        output_dir / "champion_probabilities_given_playoff_qualification.csv",
    )

    metadata = {
        "model": "path_a_poisson_dynamic_ratings",
        "simulations": simulations,
        "seed": seed,
        "feature_columns": FEATURE_COLUMNS,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caminho A: Poisson + rating dinâmico + Monte Carlo para Copa 2026.",
    )
    parser.add_argument("--historical-matches", required=True, help="CSV com resultados históricos.")
    parser.add_argument("--schedule", required=True, help="CSV com tabela da Copa.")
    parser.add_argument("--rankings", default=None, help="CSV com ranking FIFA histórico.")
    parser.add_argument("--odds", default=None, help="CSV opcional com odds próprias.")
    parser.add_argument("--playoff-schedule", default=None, help="CSV opcional com jogos da repescagem.")
    parser.add_argument("--simulations", type=int, default=1000, help="Número de simulações Monte Carlo.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--output-dir", default="outputs/path_a", help="Diretório de saída.")
    return parser.parse_args()

    