"""Build simple baseline features from prediction instances.

This module implements the first feature-engineering step for the project.
It augments prediction instances with baseline features derived only from the
current class context and booking snapshot available at prediction time.
The function is intentionally incremental: it preserves all existing columns
so later feature groups can continue working from the same evolving DataFrame.
That keeps the pipeline readable while avoiding duplication of upstream data.
"""

from __future__ import annotations

import pandas as pd


def add_baseline_features(prediction_instances: pd.DataFrame) -> pd.DataFrame:
    """Add baseline features to the prediction-instance table.

    Parameters
    ----------
    prediction_instances : pd.DataFrame
        Prediction-instance table from Notebook 02. Each row represents one
        class observed at one prediction horizon and includes the booking
        snapshot available at prediction time.

    Returns
    -------
    pd.DataFrame
        Copy of ``prediction_instances`` with additional baseline feature
        columns. Existing metadata, target, and raw booking-list columns are
        preserved for later feature-engineering steps.
    """

    result = prediction_instances.copy()

    result["attendance_count"] = result["attendance_list"].map(len)
    result["available_spots"] = result["capacity"] - result["attendance_count"]
    result["occupancy_rate"] = result["attendance_count"] / result["capacity"]
    result["is_full"] = result["attendance_count"] >= result["capacity"]
    result["has_waiting_list"] = result["waiting_list_length"] > 0
    result["weekday"] = result["class_start"].dt.day_name()
    result["class_hour"] = result["class_start"].dt.hour

    return result
