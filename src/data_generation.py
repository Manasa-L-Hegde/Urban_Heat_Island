"""Dataset generation entrypoints for the Urban Heat Island demo.

Purpose:
- Create and persist a synthetic ward-level CSV used across the application.

What it does in the project:
- Calls the synthetic data generator to produce a reproducible dataset of wards
  and writes it to `data/ward_data.csv` so downstream processing modules can load it.

Used by:
- Local development and CI to produce a demo dataset consumed by the app.

Input → Process → Output:
- Input: generator parameters (`num_wards`, `seed`) → Process: synthesize rows →
- Output: CSV file at `data/ward_data.csv`.
"""

from __future__ import annotations

from pathlib import Path

from data.synthetic_data import create_city_ward_dataset, save_city_ward_dataset


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ward_data.csv"


def generate_ward_data(num_wards: int = 400, seed: int = 42) -> "pd.DataFrame":
    """Generate a synthetic ward-level DataFrame.

    Parameters:
    - num_wards (int): Number of wards to synthesize.
    - seed (int): RNG seed for reproducibility.

    Returns:
    - pd.DataFrame: Generated ward dataset (unsaved).

    Delegates to `data.synthetic_data.create_city_ward_dataset` for generation.
    """
    return create_city_ward_dataset(num_wards=num_wards, seed=seed)


def main() -> None:
    """Generate the dataset and save it to disk at `DATA_PATH`.

    This is the CLI entrypoint when running the module directly.
    """
    df = generate_ward_data()
    save_city_ward_dataset(df, DATA_PATH)
    print(f"Saved ward dataset to {DATA_PATH}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
