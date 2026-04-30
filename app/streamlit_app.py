from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from visualization.folium_map import create_heat_map


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "outputs" / "ward_heat_processed.csv"


st.set_page_config(page_title="Urban Heat Island Hyperlocal Mapping", page_icon="🌡️", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def render_metric_row(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wards", len(df))
    col2.metric("Average Heat Score", f"{df['heat_score'].mean():.1f}")
    col3.metric("Average Vulnerability", f"{df['vulnerability_index'].mean():.1f}")
    col4.metric("Extreme Risk Wards", int((df['risk_level'] == 'Extreme').sum()))


st.title("Urban Heat Island Hyperlocal Mapping")
st.caption("Ward-level hotspot mapping using synthetic satellite thermal signals, weather station data, and building density.")

if not DATA_PATH.exists():
    st.warning("No generated dataset found yet. Run `python run_generate_data.py` first.")
    st.stop()

heat_df = load_data()

render_metric_row(heat_df)

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Interactive Heat Map")
    map_obj = create_heat_map(heat_df)
    components.html(map_obj._repr_html_(), height=760, scrolling=True)

with right:
    st.subheader("Ward Advisory Dashboard")
    ward_options = heat_df.sort_values("priority_rank", ascending=False)["ward_name"].tolist()
    selected_ward = st.selectbox("Select a ward", ward_options)
    ward_row = heat_df.loc[heat_df["ward_name"] == selected_ward].iloc[0]

    st.metric("Heat Score", f"{ward_row['heat_score']:.1f}")
    st.metric("Vulnerability Index", f"{ward_row['vulnerability_index']:.1f}")
    st.metric("Risk Level", ward_row["risk_level"])

    st.info(ward_row["heat_advisory"])
    st.write("**Recommendations**")
    st.write(ward_row["recommendations"])

    st.write("**Input signals**")
    signal_df = pd.DataFrame(
        {
            "Signal": [
                "Land Surface Temp (C)",
                "Weather Air Temp (C)",
                "Building Density",
                "Tree Cover (%)",
                "Population Density",
            ],
            "Value": [
                ward_row["land_surface_temp_c"],
                ward_row["weather_air_temp_c"],
                ward_row["building_density"],
                ward_row["tree_cover_pct"],
                ward_row["population_density"],
            ],
        }
    )
    st.dataframe(signal_df, use_container_width=True, hide_index=True)

st.subheader("High Priority Wards")
st.dataframe(
    heat_df[
        [
            "ward_name",
            "heat_score",
            "vulnerability_index",
            "risk_level",
            "recommendations",
        ]
    ].head(10),
    use_container_width=True,
    hide_index=True,
)
