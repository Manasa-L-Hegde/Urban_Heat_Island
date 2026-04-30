from __future__ import annotations

import numpy as np
import pandas as pd


def min_max_scale_0_100(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.full(len(series), 50.0), index=series.index)
    return 100 * (series - min_val) / (max_val - min_val)


def add_heat_features(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "Satellite Temperature (C)",
        "Weather Station Temperature (C)",
        "Population Density (people/km²)",
        "Building Density (0-1)",
        "Green Cover (%)",
        "Elderly Population (%)",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    enriched = df.copy()
    enriched["Final_Temperature"] = (
        0.7 * enriched["Satellite Temperature (C)"]
        + 0.3 * enriched["Weather Station Temperature (C)"]
    ).round(2)

    enriched["Final_Temperature_Normalized"] = min_max_scale_0_100(enriched["Final_Temperature"]).round(2)
    enriched["Population_Density_Normalized"] = min_max_scale_0_100(
        enriched["Population Density (people/km²)"]
    ).round(2)
    enriched["Building_Density_Normalized"] = min_max_scale_0_100(
        enriched["Building Density (0-1)"]
    ).round(2)
    enriched["Green_Cover_Normalized"] = min_max_scale_0_100(enriched["Green Cover (%)"]).round(2)
    enriched["Elderly_Population_Normalized"] = min_max_scale_0_100(
        enriched["Elderly Population (%)"]
    ).round(2)

    heat_raw = (
        0.5 * enriched["Final_Temperature_Normalized"]
        + 0.2 * enriched["Population_Density_Normalized"]
        + 0.2 * enriched["Building_Density_Normalized"]
        - 0.1 * enriched["Green_Cover_Normalized"]
    )
    enriched["Heat_Index"] = min_max_scale_0_100(heat_raw).round(2)

    vulnerability_raw = enriched["Heat_Index"] + (0.2 * enriched["Elderly Population (%)"])
    enriched["Vulnerability_Index"] = min_max_scale_0_100(vulnerability_raw).round(2)

    enriched["Risk_Level"] = pd.cut(
        enriched["Vulnerability_Index"],
        bins=[-np.inf, 40, 70, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
        right=False,
    ).astype(str)

    enriched["Vulnerability_Category"] = enriched["Risk_Level"]

    enriched["Heat_Risk_Label"] = pd.cut(
        enriched["Heat_Index"],
        bins=[-np.inf, 33, 66, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    # Add real-time heat advisory column based on Heat_Index thresholds
    enriched["Heat_Advisory"] = pd.cut(
        enriched["Heat_Index"],
        bins=[-np.inf, 60, 80, np.inf],
        labels=["Safe Conditions", "Moderate Heat Warning", "Extreme Heat Alert"],
        include_lowest=True,
        right=False,
    ).astype(str)

    return enriched
