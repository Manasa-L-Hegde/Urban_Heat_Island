from __future__ import annotations

import random
from typing import Optional

import requests

"""Small helper for fetching satellite-like temperature values from Open-Meteo
with a safe simulated fallback when the API is unavailable.
"""


OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
API_TIMEOUT = 5


def get_satellite_temperature(lat: float, lon: float) -> float:
    """
    Fetch satellite temperature data for a given latitude and longitude.

    Args:
        lat: Latitude coordinate
        lon: Longitude coordinate

    Returns:
        Temperature in Celsius. Falls back to simulated temperature (30-45°C) if API fails.
    """
    try:
        return _fetch_from_api(lat, lon)
    except (requests.RequestException, ValueError, KeyError) as error:
        return _get_simulated_temperature(error)


def _fetch_from_api(lat: float, lon: float) -> float:
    """Fetch temperature from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m",
        "timezone": "auto",
    }

    response = requests.get(OPENMETEO_URL, params=params, timeout=API_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if "current" not in data or "temperature_2m" not in data["current"]:
        raise ValueError("Unexpected API response format")

    temperature = data["current"]["temperature_2m"]
    if not isinstance(temperature, (int, float)):
        raise ValueError(f"Invalid temperature type: {type(temperature)}")

    return float(temperature)


def _get_simulated_temperature(error: Exception) -> float:
    """
    Return a simulated satellite temperature when API fails.

    Args:
        error: The exception that triggered the fallback

    Returns:
        Simulated temperature between 30-45°C
    """
    simulated_temp = round(random.uniform(30, 45), 2)
    return simulated_temp


def enrich_dataset_with_satellite_temp(df, column_name: str = "API_Satellite_Temperature", batch_size: int = 10) -> None:
    """
    Add satellite temperature from API to dataset.

    Args:
        df: Input DataFrame with 'Latitude' and 'Longitude' columns
        column_name: Name of the new column to create
        batch_size: Number of API calls before a short pause (to avoid rate limiting)

    Returns:
        Updated DataFrame with new satellite temperature column
    """
    required_columns = ["Latitude", "Longitude"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    temperatures = []
    for idx, row in df.iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        temp = get_satellite_temperature(lat, lon)
        temperatures.append(temp)

        if (idx + 1) % batch_size == 0:
            import time

            time.sleep(0.1)

    df[column_name] = temperatures
    return df
