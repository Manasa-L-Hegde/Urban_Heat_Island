from pathlib import Path

from data.synthetic_data import build_synthetic_heat_dataset, save_synthetic_outputs
from processing.heat_pipeline import prepare_heat_pipeline


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def main() -> None:
    ward_data, station_data = build_synthetic_heat_dataset()
    processed_data = prepare_heat_pipeline(ward_data, station_data)
    save_synthetic_outputs(processed_data, station_data, OUTPUT_DIR)
    print(f"Generated synthetic data in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
