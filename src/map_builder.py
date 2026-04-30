"""Map helpers for the Urban Heat Island dashboard.

This module builds a clean Folium map from ward-level heat metrics.
"""

from __future__ import annotations

import html

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Ensure that all required columns are present in `df`.

    Parameters:
    - df (pd.DataFrame): DataFrame to validate.
    - columns (list[str]): Required column names.

    Raises:
    - KeyError: If any required column is missing.
    """
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _color_for_heat_index(heat_index: float) -> str:
    """Map a heat index score to a simple severity color."""
    if heat_index < 40:
        return "green"
    if heat_index <= 70:
        return "orange"
    return "red"


def build_map(df: pd.DataFrame, center=None, selected_ward=None) -> folium.Map:
    """Build a clean interactive city heat map from ward-level data.

    Parameters:
    - df (pd.DataFrame): Ward-level dataset with coordinates and heat metrics.
    - center (tuple | list | None): Optional map center override as `(lat, lon)`.
    - selected_ward (str | None): Ward to highlight and center on if found.

    Returns:
    - folium.Map: Interactive map with heat layer and ward markers.

    Expected columns:
    - Ward Name
    - Latitude
    - Longitude
    - Heat_Index
    - Vulnerability_Index
    - AI_Predicted_Risk
    - Recommendations
    """
    required_columns = [
        "Ward Name",
        "Latitude",
        "Longitude",
        "Heat_Index",
    ]

    # Optional columns get safe defaults so the map still renders on partial data.
    optional_defaults = {
        "Vulnerability_Index": 0.0,
        "AI_Predicted_Risk": "Unknown",
        "Recommendations": "No recommendation available",
    }

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Populate any missing optional columns before numeric conversion.
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    map_df = df.copy()
    map_df["Latitude"] = pd.to_numeric(map_df["Latitude"], errors="coerce")
    map_df["Longitude"] = pd.to_numeric(map_df["Longitude"], errors="coerce")
    map_df["Heat_Index"] = pd.to_numeric(map_df["Heat_Index"], errors="coerce")
    map_df["Vulnerability_Index"] = pd.to_numeric(map_df["Vulnerability_Index"], errors="coerce")

    map_df = map_df.dropna(subset=["Latitude", "Longitude", "Heat_Index"])
    if map_df.empty:
        raise ValueError("No valid latitude/longitude/heat index rows found to build the map.")

    # If a ward is selected, try to center and zoom to that ward.
    zoom_start = 12
    if selected_ward is not None:
        try:
            match = map_df[map_df["Ward Name"].astype(str).str.strip().str.lower() == str(selected_ward).strip().lower()]
            if not match.empty:
                center_lat = float(match.iloc[0]["Latitude"])
                center_lon = float(match.iloc[0]["Longitude"])
                zoom_start = 15
            else:
                # Fallback to the provided center or the overall mean coordinates.
                if center is not None:
                    center_lat = float(center[0])
                    center_lon = float(center[1])
                else:
                    center_lat = float(map_df["Latitude"].mean())
                    center_lon = float(map_df["Longitude"].mean())
        except Exception:
            center_lat = float(map_df["Latitude"].mean())
            center_lon = float(map_df["Longitude"].mean())
    else:
        if center is not None:
            try:
                center_lat = float(center[0])
                center_lon = float(center[1])
            except Exception:
                center_lat = float(map_df["Latitude"].mean())
                center_lon = float(map_df["Longitude"].mean())
        else:
            center_lat = float(map_df["Latitude"].mean())
            center_lon = float(map_df["Longitude"].mean())

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="cartodbpositron",
        control_scale=True,
        prefer_canvas=True,
    )

    sample_df = map_df.sample(n=min(len(map_df), 150), random_state=42)

    heat_data = [
        [float(row["Latitude"]), float(row["Longitude"]), max(0.1, min(1.0, float(row["Heat_Index"]) / 100.0))]
        for _, row in sample_df.iterrows()
    ]
    HeatMap(heat_data, radius=10, blur=16, min_opacity=0.25).add_to(m)

    marker_cluster = MarkerCluster(name="Ward markers").add_to(m)

    for _, row in sample_df.iterrows():
        ward_name = str(row["Ward Name"]).strip()
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        heat_index = float(row["Heat_Index"])

        # Draw the selected ward as a prominent marker so it stands out.
        if selected_ward is not None and ward_name.strip().lower() == str(selected_ward).strip().lower():
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.9,
                popup=folium.Popup(f"<b>{html.escape(ward_name)} (Selected)</b>", max_width=320),
                tooltip=folium.Tooltip(html.escape(ward_name), sticky=True),
            ).add_to(marker_cluster)
            continue

        color = _color_for_heat_index(heat_index)
        vulnerability = float(row.get("Vulnerability_Index", 0.0))

        popup_html = f"""
        <div style="min-width:200px;font-family:Arial,sans-serif;">
            <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:6px;">{html.escape(ward_name)}</div>
            <div style="font-size:12px;margin:2px 0;"><b>Heat Index:</b> {heat_index:.2f}</div>
            <div style="font-size:12px;margin:2px 0;"><b>Vulnerability:</b> {vulnerability:.2f}</div>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=folium.Tooltip(html.escape(ward_name), sticky=True),
        ).add_to(marker_cluster)

    return m