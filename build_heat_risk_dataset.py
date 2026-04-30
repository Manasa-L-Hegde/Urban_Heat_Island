from pathlib import Path

from data.synthetic_data import create_city_ward_dataset, save_city_ward_dataset
from models.heat_risk_model import add_ai_predictions, train_heat_risk_model
from processing.feature_engineering import add_heat_features


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / "outputs" / "city_ward_dataset.csv"


def build_dataset(num_wards: int = 400, seed: int = 42):
    raw_df = create_city_ward_dataset(num_wards=num_wards, seed=seed)
    processed_df = add_heat_features(raw_df)
    model_result = train_heat_risk_model(processed_df, random_state=seed)
    final_df = add_ai_predictions(processed_df, model_result)
    save_city_ward_dataset(final_df, OUTPUT_PATH)
    return final_df, model_result.accuracy


def main() -> None:
    df, accuracy = build_dataset()
    print(df.head(10).to_string(index=False))
    print(f"\nRows: {len(df)}")
    print(f"Model accuracy: {accuracy:.3f}")
    print(f"Saved CSV to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
