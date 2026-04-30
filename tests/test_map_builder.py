import os
import sys
import pandas as pd

# Ensure test can import package modules by adding project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.map_builder import build_map
import folium


def test_build_map_returns_folium_map():
    df = pd.DataFrame([
        {
            "Ward Name": "A",
            "Latitude": 12.97,
            "Longitude": 77.59,
            "Heat_Index": 55.0,
            "Vulnerability_Index": 40.0,
            "AI_Predicted_Risk": "Medium",
            "Recommendations": "Plant trees",
        },
        {
            "Ward Name": "B",
            "Latitude": 12.98,
            "Longitude": 77.60,
            "Heat_Index": 65.0,
            "Vulnerability_Index": 45.0,
            "AI_Predicted_Risk": "High",
            "Recommendations": "Install cool roofs",
        },
    ])
    m = build_map(df)
    assert isinstance(m, folium.Map)


def test_build_map_with_selected_ward():
    df = pd.DataFrame([
        {
            "Ward Name": "A",
            "Latitude": 12.97,
            "Longitude": 77.59,
            "Heat_Index": 55.0,
        },
        {
            "Ward Name": "B",
            "Latitude": 12.98,
            "Longitude": 77.60,
            "Heat_Index": 65.0,
        },
    ])
    # Ensure providing selected_ward doesn't raise and returns a map
    m = build_map(df, selected_ward="A")
    assert isinstance(m, folium.Map)
