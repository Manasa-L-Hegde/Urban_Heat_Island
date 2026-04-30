from pathlib import Path

from build_heat_risk_dataset import build_dataset


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "outputs" / "city_ward_dataset.csv"


def main() -> None:
    df, accuracy = build_dataset(num_wards=400, seed=42)
    print(df.head(10).to_string(index=False))
    print(f"\nRows: {len(df)}")
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"\nSaved CSV to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
