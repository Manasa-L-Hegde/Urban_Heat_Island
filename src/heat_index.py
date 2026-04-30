"""Heat index assembly for the dashboard.

Purpose:
- Provide a small wrapper that prepares the normalized heat and vulnerability
    metrics used by the UI and map builder.

Input → Process → Output:
- Input: DataFrame with raw temperature and demographic columns.
- Process: delegates to `processing.feature_engineering.add_heat_features`.
- Output: DataFrame with normalized features and labels added.
"""

from __future__ import annotations

import pandas as pd

from processing.feature_engineering import add_heat_features


def build_heat_index(df: pd.DataFrame) -> pd.DataFrame:
        """Generate heat-related features and labels.

        Parameters
        - df (pd.DataFrame): Input ward-level DataFrame with raw inputs.

        Returns
        - pd.DataFrame: Enriched DataFrame with `Final_Temperature`, `Heat_Index`,
            `Vulnerability_Index`, and categorical risk labels.

        This function is a thin wrapper around `add_heat_features` to keep
        high-level pipeline calls readable.
        """
        # Delegate the heavy lifting to the processing module
        return add_heat_features(df)
