from pathlib import Path

import pandas as pd

from src.temperature_calculation import compute_final_temperature


INPUT_PATH = Path(__file__).resolve().parent / "data" / "ward_data_with_api_satellite_temp.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "ward_data_with_final_temperature.csv"


def main() -> None:
    print(f"Loading enriched dataset from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    print(f"Computing Final_Temperature using vectorized Pandas operations...")
    print("  Formula: Final_Temperature = (0.7 * Satellite_Temperature) + (0.3 * Weather_Temperature)\n")

    df_with_final_temp = compute_final_temperature(
        df,
        satellite_col="API_Satellite_Temperature",
        weather_col="Weather Station Temperature (C)",
        output_col="Final_Temperature",
    )

    df_with_final_temp.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved dataset with Final_Temperature to {OUTPUT_PATH}")
    print(f"\nDataset shape: {df_with_final_temp.shape}")
    print(f"\nFirst 10 rows with temperature calculations:")
    print(
        df_with_final_temp[
            ["Ward Name", "API_Satellite_Temperature", "Weather Station Temperature (C)", "Final_Temperature"]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nFinal_Temperature statistics:")
    print(df_with_final_temp["Final_Temperature"].describe().round(2))


if __name__ == "__main__":
    main()
