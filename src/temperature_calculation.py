from __future__ import annotations

import pandas as pd

"""Utilities to compute vectorized temperature combinations.

Exports `compute_final_temperature` which combines satellite and weather readings.
"""


def compute_final_temperature(
    df: pd.DataFrame,
    satellite_col: str = "API_Satellite_Temperature",
    weather_col: str = "Weather Station Temperature (C)",
    output_col: str = "Final_Temperature",
) -> pd.DataFrame:
    """
    Compute final temperature using vectorized Pandas operations.

    Formula: Final_Temperature = (0.7 * Satellite_Temperature) + (0.3 * Weather_Temperature)

    Args:
        df: Input DataFrame
        satellite_col: Name of satellite temperature column
        weather_col: Name of weather station temperature column
        output_col: Name of output column to create

    Returns:
        DataFrame with new Final_Temperature column
    """
    required_columns = [satellite_col, weather_col]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    result = df.copy()
    result[output_col] = (0.7 * result[satellite_col]) + (0.3 * result[weather_col])
    result[output_col] = result[output_col].round(2)

    return result
