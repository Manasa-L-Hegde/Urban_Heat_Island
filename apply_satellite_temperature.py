from pathlib import Path
import time

import pandas as pd

from src.satellite_api import get_satellite_temperature


DATA_PATH = Path(__file__).resolve().parent / "data" / "ward_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "ward_data_with_api_satellite_temp.csv"
DEMO_SIZE = 50


def main() -> None:
    print(f"Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print(f"Fetching satellite temperatures for first {DEMO_SIZE} wards (demo subset)...")
    print("This uses real Open-Meteo API data with graceful fallback to simulated temperatures on failure.\n")

    temperatures = []
    start_time = time.time()
    
    for idx, row in df.head(DEMO_SIZE).iterrows():
        lat = float(row["Latitude"])
        lon = float(row["Longitude"])
        temp = get_satellite_temperature(lat, lon)
        temperatures.append(temp)
        
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Progress: {idx + 1}/{DEMO_SIZE} wards ({100 * (idx + 1) / DEMO_SIZE:.1f}%) - {elapsed:.1f}s elapsed")
            time.sleep(0.1)

    df["API_Satellite_Temperature"] = pd.Series(temperatures, index=df.head(DEMO_SIZE).index)
    
    df.to_csv(OUTPUT_PATH, index=False)
    elapsed = time.time() - start_time
    print(f"\nUpdated dataset saved to {OUTPUT_PATH}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"\nFirst 10 wards with new API satellite temperature:")
    print(df[["Ward Name", "Latitude", "Longitude", "API_Satellite_Temperature"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
