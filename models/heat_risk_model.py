"""Heat risk modeling utilities.

Purpose:
- Train and apply a simple supervised model to predict ward-level heat risk labels.

What it does in the project:
- Exposes a training function and a prediction-enrichment helper used by the
    dashboard and advisor components to produce AI-predicted risk labels for wards.

Used by:
- `src/heat_risk_advisor.py` (generates AI-backed recommendations)
- `app/app.py` (optional model-run for demo/insights)

Input → Process → Output:
- Input: ward-level DataFrame with normalized features and a target label.
- Process: encode labels, split, train a RandomForest, evaluate accuracy.
- Output: `HeatRiskModelResult` containing trained model, encoder, and accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


FEATURE_COLUMNS = [
    "Final_Temperature_Normalized",
    "Population_Density_Normalized",
    "Building_Density_Normalized",
    "Green_Cover_Normalized",
]
TARGET_COLUMN = "Heat_Risk_Label"


@dataclass(frozen=True)
class HeatRiskModelResult:
    """Container for a trained heat risk classifier and metadata.

    Attributes:
    - model: Trained `RandomForestClassifier` instance.
    - label_encoder: `LabelEncoder` used to encode/decode target labels.
    - accuracy: Accuracy on the held-out test set (0.0 - 1.0).
    """

    model: RandomForestClassifier
    label_encoder: LabelEncoder
    accuracy: float


def train_heat_risk_model(df: pd.DataFrame, random_state: int = 42) -> HeatRiskModelResult:
    """Train a RandomForest model to predict `TARGET_COLUMN` from features.

    Parameters
    - df (pd.DataFrame): DataFrame containing feature columns and the target column.
    - random_state (int): Seed for reproducible train/test split and model.

    Returns
    - HeatRiskModelResult: dataclass with trained model, label encoder, and test accuracy.

    Logic:
    1. Validate required columns are present.
    2. Extract features and the target label.
    3. Encode categorical target labels to integers.
    4. Split data; stratify only when there are multiple classes.
    5. Train a RandomForest with balanced class weights and evaluate accuracy.
    """

    # Ensure input DataFrame contains required columns
    missing_columns = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns for model training: {missing_columns}")

    features = df[FEATURE_COLUMNS]
    target = df[TARGET_COLUMN]

    # Convert string/label target to integer encoding for sklearn
    label_encoder = LabelEncoder()
    encoded_target = label_encoder.fit_transform(target)

    # Only stratify split when there is more than one class present
    stratify_target = encoded_target if len(set(encoded_target)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        encoded_target,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify_target,
    )

    # Configure a reasonably small RandomForest for the demo (balanced classes)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    # Evaluate on the held-out test set
    accuracy = accuracy_score(y_test, model.predict(x_test))

    return HeatRiskModelResult(model=model, label_encoder=label_encoder, accuracy=accuracy)


def add_ai_predictions(df: pd.DataFrame, result: HeatRiskModelResult) -> pd.DataFrame:
        """Apply a trained model to a DataFrame and add `AI_Predicted_Risk` column.

        Parameters
        - df (pd.DataFrame): DataFrame containing at least `FEATURE_COLUMNS`.
        - result (HeatRiskModelResult): Trained model container returned by `train_heat_risk_model`.

        Returns
        - pd.DataFrame: A copy of `df` with a new `AI_Predicted_Risk` column containing
            decoded label strings.

        Notes:
        - This helper does not validate all input columns beyond using `FEATURE_COLUMNS`.
        - Predictions are produced using the provided `result.model` and decoded
            back to original label strings with `result.label_encoder`.
        """

        # Predict integer labels then convert back to original string labels
        predicted_labels = result.label_encoder.inverse_transform(result.model.predict(df[FEATURE_COLUMNS]))
        enriched = df.copy()
        enriched["AI_Predicted_Risk"] = predicted_labels
        return enriched
