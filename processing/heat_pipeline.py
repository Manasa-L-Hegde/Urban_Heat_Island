from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _scale_0_100(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)


def calculate_heat_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    satellite_term = _zscore(scored["land_surface_temp_c"])
    weather_term = _zscore(scored["weather_air_temp_c"])
    building_term = _zscore(scored["building_density"])
    greenery_term = _zscore(100 - scored["tree_cover_pct"])

    raw_heat = (
        0.35 * satellite_term
        + 0.25 * weather_term
        + 0.20 * building_term
        + 0.20 * greenery_term
    )
    scored["heat_score"] = _scale_0_100(raw_heat).round(2)
    return scored


def calculate_vulnerability_index(df: pd.DataFrame) -> pd.DataFrame:
    vulnerable = df.copy()
    pop_term = _zscore(vulnerable["population_density"])
    elderly_term = _zscore(vulnerable["elderly_share_pct"])
    greenery_term = _zscore(100 - vulnerable["tree_cover_pct"])
    income_term = _zscore(1 - vulnerable["income_index"])

    raw_vulnerability = (
        0.35 * pop_term
        + 0.25 * elderly_term
        + 0.20 * greenery_term
        + 0.20 * income_term
    )
    vulnerable["vulnerability_index"] = _scale_0_100(raw_vulnerability).round(2)
    return vulnerable


def classify_risk_level(heat_score: float, vulnerability_index: float) -> str:
    composite = 0.6 * heat_score + 0.4 * vulnerability_index
    if composite >= 75:
        return "Extreme"
    if composite >= 60:
        return "High"
    if composite >= 40:
        return "Moderate"
    return "Low"


def recommend_actions(row: pd.Series) -> str:
    recommendations: list[str] = []

    if row["tree_cover_pct"] < 18 and row["heat_score"] >= 55:
        recommendations.append("Increase tree planting and shade corridors")
    if row["building_density"] >= 60 or row["impervious_surface"] >= 65:
        recommendations.append("Promote reflective roofs and cool paving")
    if row["vulnerability_index"] >= 65:
        recommendations.append("Prioritize cooling shelters and drinking water points")
    if row["weather_humidity_pct"] <= 35 and row["weather_air_temp_c"] >= 35:
        recommendations.append("Issue heat alerts for outdoor workers and vulnerable residents")
    if not recommendations:
        recommendations.append("Maintain current green cover and monitor weekly")

    return "; ".join(recommendations)


def build_realtime_advisory(row: pd.Series) -> str:
    if row["heat_score"] >= 75 and row["weather_air_temp_c"] >= 36:
        return f"Severe heat advisory for {row['ward_name']}: limit outdoor activity, hydrate, and activate cooling support."
    if row["heat_score"] >= 60:
        return f"Heat watch for {row['ward_name']}: recommend shade, hydration, and schedule changes for outdoor work."
    return f"Routine monitoring for {row['ward_name']}: conditions are currently manageable."


def prepare_heat_pipeline(ward_df: pd.DataFrame, station_df: pd.DataFrame | None = None) -> pd.DataFrame:
    processed = calculate_heat_score(ward_df)
    processed = calculate_vulnerability_index(processed)
    processed["risk_level"] = processed.apply(
        lambda row: classify_risk_level(row["heat_score"], row["vulnerability_index"]), axis=1
    )
    processed["recommendations"] = processed.apply(recommend_actions, axis=1)
    processed["heat_advisory"] = processed.apply(build_realtime_advisory, axis=1)
    processed["priority_rank"] = processed["heat_score"] * 0.7 + processed["vulnerability_index"] * 0.3
    processed = processed.sort_values(["priority_rank", "heat_score"], ascending=False).reset_index(drop=True)
    return processed
