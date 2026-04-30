"""Data fusion helpers for combining temperature sources.

Purpose:
- Provide a reproducible rule to merge satellite and weather-station temperatures
  into a single `Final_Temperature` used across downstream features.

Input → Process → Output:
- Input: DataFrame with satellite and weather temperature columns.
- Process: weighted linear combination (0.7 satellite, 0.3 weather).
- Output: DataFrame with `Final_Temperature` column appended.
"""

from __future__ import annotations

import pandas as pd


SATELLITE_COLUMN = "Satellite Temperature (C)"
WEATHER_COLUMN = "Weather Station Temperature (C)"
FINAL_COLUMN = "Final_Temperature"


def fuse_temperatures(df: pd.DataFrame) -> pd.DataFrame:
    """Combine satellite and weather temperatures into a final temperature.

    Parameters
    - df (pd.DataFrame): Input DataFrame containing satellite and weather temp columns.

    Returns
    - pd.DataFrame: Copy of `df` with `Final_Temperature` computed.

    Calculation: `Final_Temperature = 0.7 * Satellite + 0.3 * Weather`.
    """
    required_columns = [SATELLITE_COLUMN, WEATHER_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    fused = df.copy()
    # Weighted average: give more weight to satellite temperature
    fused[FINAL_COLUMN] = (0.7 * fused[SATELLITE_COLUMN]) + (0.3 * fused[WEATHER_COLUMN])
    return fused
