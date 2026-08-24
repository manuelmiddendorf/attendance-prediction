"""Add features describing the current class at prediction time.

This module implements the final incremental feature group in Notebook 03.
It combines demand relative to capacity with a small, leakage-safe summary
of prior classes for the same course and instructor.
All calculations return a copy of the existing feature table so earlier
pipeline stages and their target values remain unchanged.
"""

from __future__ import annotations

import pandas as pd


def add_class_context_features(
    features: pd.DataFrame,
    attendance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add current class-context features to the incremental feature table.

    Parameters
    ----------
    features : pd.DataFrame
        Current feature table with canonical baseline counts, ``capacity``,
        ``course``, ``instructor``, and ``prediction_time``.
    attendance_history : pd.DataFrame
        Canonical final attendance history. Prior classes of the same
        course-instructor combination contribute to the 90-day historical
        features.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with total demand and course-instructor history
        features added.
    """

    result = features.copy()

    result["total_demand_count"] = (
        result["attendance_count"] + result["waiting_list_length"]
    )

    valid_capacity = result["capacity"].notna() & result["capacity"].gt(0)
    result["total_demand_fill_ratio"] = (
        result["total_demand_count"] / result["capacity"]
    ).where(valid_capacity)

    history = attendance_history.copy()
    history["final_attendance_count"] = history["attendance_list"].map(len)
    attendance_means = []
    history_counts = []

    for row in result.itertuples(index=False):
        window_start = row.prediction_time - pd.Timedelta(days=90)
        combination_history = history.loc[
            (history["course"] == row.course)
            & (history["instructor"] == row.instructor)
            & (history["class_start"] >= window_start)
            & (history["class_start"] < row.prediction_time)
        ]

        attendance_means.append(combination_history["final_attendance_count"].mean())
        history_counts.append(len(combination_history))

    result["course_instructor_attendance_mean_90d"] = attendance_means
    result["course_instructor_history_count_90d"] = history_counts

    return result
