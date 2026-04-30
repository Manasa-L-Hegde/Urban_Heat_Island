from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CityConfig:
    city_name: str = "Demo City"
    center_lat: float = 28.6139
    center_lon: float = 77.2090
    ward_rows: int = 4
    ward_cols: int = 5
    ward_spacing: float = 0.018


CITY = CityConfig()


def _min_max_scale_0_100(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)


def _grid_offsets(rows: int, cols: int, spacing: float) -> list[tuple[float, float]]:
    lat_offsets = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) * spacing
    lon_offsets = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) * spacing
    return [(float(lat), float(lon)) for lat in lat_offsets for lon in lon_offsets]


def generate_ward_base_data(num_wards: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    offsets = _grid_offsets(CITY.ward_rows, CITY.ward_cols, CITY.ward_spacing)[:num_wards]

    wards = []
    for idx, (lat_offset, lon_offset) in enumerate(offsets, start=1):
        tree_cover = np.clip(rng.normal(28 - idx * 0.5, 8), 5, 55)
        building_density = np.clip(rng.normal(40 + idx * 1.7, 12), 10, 95)
        population_density = np.clip(rng.normal(6500 + idx * 220, 1400), 1200, 18000)
        impervious_surface = np.clip(0.45 * building_density + 0.35 * (100 - tree_cover) + rng.normal(0, 4), 15, 98)
        elderly_share = np.clip(rng.normal(10 + idx * 0.15, 2), 5, 24)
        income_index = np.clip(rng.normal(0.55 - idx * 0.01, 0.08), 0.2, 0.85)
        lat = CITY.center_lat + lat_offset
        lon = CITY.center_lon + lon_offset

        wards.append(
            {
                "ward_id": f"W{idx:02d}",
                "ward_name": f"Ward {idx:02d}",
                "city": CITY.city_name,
                "lat": lat,
                "lon": lon,
                "population_density": round(float(population_density), 2),
                "building_density": round(float(building_density), 2),
                "impervious_surface": round(float(impervious_surface), 2),
                "tree_cover_pct": round(float(tree_cover), 2),
                "elderly_share_pct": round(float(elderly_share), 2),
                "income_index": round(float(income_index), 3),
            }
        )

    return pd.DataFrame(wards)


def simulate_satellite_thermal_data(ward_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    satellite = ward_df[["ward_id", "lat", "lon", "tree_cover_pct", "building_density", "impervious_surface"]].copy()

    base_temp = 31.0
    satellite_anomaly = (
        0.08 * satellite["building_density"]
        + 0.05 * satellite["impervious_surface"]
        - 0.06 * satellite["tree_cover_pct"]
        + rng.normal(0, 1.2, len(satellite))
    )

    satellite["land_surface_temp_c"] = (base_temp + satellite_anomaly).round(2)
    satellite["thermal_anomaly_c"] = satellite_anomaly.round(2)
    satellite["satellite_quality"] = rng.choice(["good", "usable", "cloud_filtered"], size=len(satellite), p=[0.45, 0.35, 0.20])
    satellite["satellite_pass"] = pd.Timestamp.utcnow().normalize().strftime("%Y-%m-%d")
    return satellite.drop(columns=["tree_cover_pct", "building_density", "impervious_surface"])


def simulate_weather_station_data(num_stations: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 200)
    station_offsets = np.linspace(-0.03, 0.03, num_stations)

    stations = []
    for idx, offset in enumerate(station_offsets, start=1):
        air_temp = np.clip(rng.normal(33.5 + idx * 0.3, 1.1), 28, 42)
        humidity = np.clip(rng.normal(48 - idx * 1.5, 6), 18, 78)
        wind_speed = np.clip(rng.normal(2.2 + idx * 0.1, 0.8), 0.2, 6.5)
        stations.append(
            {
                "station_id": f"S{idx:02d}",
                "station_name": f"Station {idx:02d}",
                "lat": CITY.center_lat + offset,
                "lon": CITY.center_lon - offset,
                "air_temp_c": round(float(air_temp), 2),
                "humidity_pct": round(float(humidity), 2),
                "wind_speed_mps": round(float(wind_speed), 2),
                "observation_time": pd.Timestamp.utcnow().floor("h").strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    return pd.DataFrame(stations)


def assign_nearest_station(ward_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    assigned = ward_df.copy()

    station_points = station_df[["lat", "lon", "air_temp_c", "humidity_pct", "wind_speed_mps"]].to_numpy(dtype=float)
    station_ids = station_df["station_id"].tolist()
    nearest_station_ids = []
    nearest_air_temp = []
    nearest_humidity = []
    nearest_wind = []
    nearest_station_distance = []

    for _, ward in assigned.iterrows():
        ward_point = np.array([ward["lat"], ward["lon"]])
        distances = np.sqrt((station_points[:, 0] - ward_point[0]) ** 2 + (station_points[:, 1] - ward_point[1]) ** 2)
        nearest_index = int(np.argmin(distances))
        nearest_station_ids.append(station_ids[nearest_index])
        nearest_air_temp.append(float(station_points[nearest_index, 2]))
        nearest_humidity.append(float(station_points[nearest_index, 3]))
        nearest_wind.append(float(station_points[nearest_index, 4]))
        nearest_station_distance.append(float(distances[nearest_index]))

    assigned["nearest_station_id"] = nearest_station_ids
    assigned["weather_air_temp_c"] = np.round(nearest_air_temp, 2)
    assigned["weather_humidity_pct"] = np.round(nearest_humidity, 2)
    assigned["weather_wind_speed_mps"] = np.round(nearest_wind, 2)
    assigned["station_distance_deg"] = np.round(nearest_station_distance, 5)
    return assigned


def build_synthetic_heat_dataset(num_wards: int = 20, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ward_base = generate_ward_base_data(num_wards=num_wards, seed=seed)
    satellite_df = simulate_satellite_thermal_data(ward_base, seed=seed)
    station_df = simulate_weather_station_data(seed=seed)

    merged = ward_base.merge(satellite_df, on=["ward_id", "lat", "lon"], how="left")
    merged = assign_nearest_station(merged, station_df)
    return merged, station_df


def save_synthetic_outputs(processed_df: pd.DataFrame, station_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_dir / "ward_heat_processed.csv", index=False)
    station_df.to_csv(output_dir / "weather_stations.csv", index=False)


def _dynamic_grid_offsets(num_wards: int) -> list[tuple[float, float]]:
    rows = int(np.ceil(np.sqrt(num_wards)))
    cols = int(np.ceil(num_wards / rows))
    spacing = 0.0048 if num_wards >= 300 else 0.0075
    return _grid_offsets(rows, cols, spacing)[:num_wards]


def create_city_ward_dataset(num_wards: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 300)
    offsets = _dynamic_grid_offsets(num_wards)

    records = []
    for idx, (lat_offset, lon_offset) in enumerate(offsets, start=1):
        radial_distance = float(np.sqrt(lat_offset**2 + lon_offset**2))
        center_factor = np.clip(1.0 - radial_distance / 0.07, 0.15, 1.0)
        zone_bias = rng.normal(0, 0.06)
        heat_pressure = np.clip(0.30 + 0.55 * center_factor + zone_bias, 0.18, 0.98)

        green_cover = np.clip(48 - 16 * heat_pressure + rng.normal(0, 6.0), 4, 70)
        building_density = np.clip(0.18 + 0.62 * heat_pressure + rng.normal(0, 0.07), 0.05, 0.98)
        population_density = np.clip(2800 + (heat_pressure * 12000) + (idx % 23) * 115 + rng.normal(0, 900), 1200, 24000)
        elderly_population = np.clip(5.0 + 10.0 * (1.0 - center_factor) + rng.normal(0, 1.7), 3.5, 26.0)

        satellite_temp = np.clip(
            30.0
            + 7.5 * building_density
            - 0.07 * green_cover
            + 1.5 * heat_pressure
            + rng.normal(0, 0.9),
            30,
            45,
        )
        weather_temp = np.clip(satellite_temp - rng.normal(0.9, 0.55), 29.5, 44.5)

        records.append(
            {
                "Ward Name": f"Ward {idx:03d}",
                "Latitude": round(float(CITY.center_lat + lat_offset), 6),
                "Longitude": round(float(CITY.center_lon + lon_offset), 6),
                "Satellite Temperature (C)": round(float(satellite_temp), 2),
                "Weather Station Temperature (C)": round(float(weather_temp), 2),
                "Population Density (people/km²)": int(round(population_density)),
                "Building Density (0-1)": round(float(building_density), 3),
                "Green Cover (%)": round(float(green_cover), 2),
                "Elderly Population (%)": round(float(elderly_population), 2),
            }
        )

    return pd.DataFrame(records)


def save_city_ward_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
