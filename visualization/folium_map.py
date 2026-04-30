from __future__ import annotations

from typing import Dict, Iterable

import folium
import pandas as pd


COLOR_STEPS = [
    (85, "#7f0000"),
    (70, "#b30000"),
    (55, "#e34a33"),
    (40, "#fc8d59"),
    (25, "#fdbb84"),
    (0, "#fee8c8"),
]


def _heat_color(value: float) -> str:
    for threshold, color in COLOR_STEPS:
        if value >= threshold:
            return color
    return "#fee8c8"


def _square_polygon(lat: float, lon: float, delta: float = 0.007) -> list[list[float]]:
    return [
        [lat - delta, lon - delta],
        [lat - delta, lon + delta],
        [lat + delta, lon + delta],
        [lat + delta, lon - delta],
        [lat - delta, lon - delta],
    ]


def build_geojson(df: pd.DataFrame) -> Dict:
    features = []
    for _, row in df.iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "ward_id": row["ward_id"],
                    "ward_name": row["ward_name"],
                    "heat_score": float(row["heat_score"]),
                    "vulnerability_index": float(row["vulnerability_index"]),
                    "risk_level": row["risk_level"],
                    "recommendations": row["recommendations"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_square_polygon(float(row["lat"]), float(row["lon"]))],
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


def create_heat_map(df: pd.DataFrame) -> folium.Map:
    center = [float(df["lat"].mean()), float(df["lon"].mean())]
    fmap = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")
    geojson_data = build_geojson(df)

    def style_function(feature: Dict) -> Dict:
        heat_score = feature["properties"]["heat_score"]
        return {
            "fillColor": _heat_color(heat_score),
            "color": "#3b3b3b",
            "weight": 1,
            "fillOpacity": 0.68,
        }

    tooltip = folium.GeoJsonTooltip(
        fields=["ward_name", "heat_score", "vulnerability_index", "risk_level", "recommendations"],
        aliases=["Ward", "Heat Score", "Vulnerability", "Risk", "Action"],
        localize=True,
        sticky=False,
        labels=True,
    )

    folium.GeoJson(
        geojson_data,
        name="Ward Heat Layer",
        style_function=style_function,
        tooltip=tooltip,
    ).add_to(fmap)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=4,
            color="#111111",
            fill=True,
            fill_color="#111111",
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{row['ward_name']}</b><br>Advisory: {row['heat_advisory']}<br>Station: {row['nearest_station_id']}",
                max_width=300,
            ),
        ).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999; background: white; padding: 12px 14px; border: 1px solid #aaa; border-radius: 8px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">
      <div style="font-weight: 700; margin-bottom: 8px;">Heat Score</div>
      <div><span style="background:#7f0000;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>85+</div>
      <div><span style="background:#b30000;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>70-84</div>
      <div><span style="background:#e34a33;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>55-69</div>
      <div><span style="background:#fc8d59;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>40-54</div>
      <div><span style="background:#fdbb84;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>25-39</div>
      <div><span style="background:#fee8c8;display:inline-block;width:14px;height:14px;margin-right:8px;"></span>0-24</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)
    return fmap
