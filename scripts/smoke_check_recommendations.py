import sys
import pathlib
import pandas as pd

# Ensure project root is on sys.path so `src` imports work when running as a script
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heat_risk_advisor import generate_green_cover_recommendations, _find_column

CSV_PATH = "data/ward_data_with_api_satellite_temp.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    out = generate_green_cover_recommendations(df)

    # Resolve display column names robustly
    display_cols = [
        _find_column(out, ["Ward Name"]),
        _find_column(out, ["Heat_Index"]),
        _find_column(out, ["Green Cover", "Green Cover (%)", "Green_Cover", "Green_Cover (%)"]),
        _find_column(out, ["Population Density", "Population_Density", "Population Density (people/km²)"]),
        _find_column(out, ["Building Density", "Building_Density", "Building Density (0-1)"]),
        "Green_Cover_Recommendation",
    ]

    pd.set_option("display.max_colwidth", 200)
    print(out[display_cols].head(10).to_string(index=False))

if __name__ == '__main__':
    main()
