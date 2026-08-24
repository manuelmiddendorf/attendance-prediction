"""Add simple historical attendance summary features.

This module extends the incremental feature table with leakage-safe summaries
computed from the full canonical attendance history.
For each prediction instance it looks only at classes that started before the
prediction time and summarizes prior attendance by course, instructor, and
studio over simple time windows.
The implementation prioritizes readability over optimization because the
project dataset is small enough for explicit row-wise logic.
"""

from __future__ import annotations

import pandas as pd


_HISTORY_WINDOWS = {
    "30d": pd.Timedelta(days=30),
    "90d": pd.Timedelta(days=90),
    "365d": pd.Timedelta(days=365),
}

_HISTORY_GROUPS = ["course", "instructor", "studio"]


def _select_past_classes(
    attendance_history: pd.DataFrame,
    prediction_time: pd.Timestamp,
) -> pd.DataFrame:
    """Return classes that started before the prediction time."""

    return attendance_history.loc[attendance_history["class_start"] < prediction_time]


def _summarize_window(
    grouped_history: pd.DataFrame,
    prediction_time: pd.Timestamp,
    window: pd.Timedelta,
) -> tuple[float, float, int]:
    """Summarize historical attendance for one temporal window.

    Parameters
    ----------
    grouped_history : pd.DataFrame
        Historical rows already filtered to one grouping value such as a single
        course, instructor, or studio.
    prediction_time : pd.Timestamp
        Time at which the prediction instance is observed.
    window : pd.Timedelta 
        Historical lookback window. 

    Returns
    -------
    tuple[float, float, int]
        Mean final attendance, standard deviation of final attendance, and the
        number of historical classes in the selected window.
    """

    window_history = grouped_history

    window_start = prediction_time - window
    window_history = window_history.loc[
       window_history["class_start"] >= window_start
    ]

    return (
        window_history["final_attendance_count"].mean(),
        window_history["final_attendance_count"].std(),
        len(window_history),
    )


def _build_group_history_features(
    attendance_history: pd.DataFrame,
    group_column: str,
    group_value: object,
    prediction_time: pd.Timestamp,
) -> dict[str, object]:
    """Build historical features for one grouping value and prediction time.

    Parameters
    ----------
    attendance_history : pd.DataFrame
        Canonical attendance history with a ``final_attendance_count`` column.
    group_column : str
        Grouping column name. Expected values are ``course``, ``instructor``,
        or ``studio``.
    group_value : object
        Value identifying the current course, instructor, or studio.
    prediction_time : pd.Timestamp
        Time at which the prediction instance is observed.

    Returns
    -------
    dict[str, object]
        Mapping from historical feature names to their calculated values for
        the requested grouping value and prediction time.
    """

    past_classes = _select_past_classes(attendance_history, prediction_time)
    grouped_history = past_classes.loc[past_classes[group_column] == group_value]

    feature_values: dict[str, object] = {}

    for window_label, window in _HISTORY_WINDOWS.items():
        mean_value, std_value, count_value = _summarize_window(
            grouped_history,
            prediction_time,
            window,
        )
        feature_values[f"{group_column}_attendance_mean_{window_label}"] = mean_value
        feature_values[f"{group_column}_attendance_std_{window_label}"] = std_value
        feature_values[f"{group_column}_attendance_count_{window_label}"] = count_value

    return feature_values


def add_historical_features(
    features: pd.DataFrame,
    attendance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe historical attendance features to the current table.

    Parameters
    ----------
    features : pd.DataFrame
        Current incremental feature table. It must contain ``course``,
        ``instructor``, ``studio``, and ``prediction_time`` columns.
    attendance_history : pd.DataFrame
        Full canonical attendance dataset. Historical attendance counts are
        derived from ``attendance_list`` and only classes before each
        prediction time are allowed to contribute.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with additional historical statistics for course,
        instructor, and studio across 30-day, 90-day, and 365-day windows.
    """

    result = features.copy()
    history = attendance_history.copy()
    history["final_attendance_count"] = history["attendance_list"].map(len)

    for group_column in _HISTORY_GROUPS:
        group_feature_rows = []

        for row in result.itertuples(index=False):
            group_feature_rows.append(
                _build_group_history_features(
                    history,
                    group_column,
                    getattr(row, group_column),
                    row.prediction_time,
                )
            )

        group_feature_frame = pd.DataFrame(group_feature_rows, index=result.index)
        result = pd.concat([result, group_feature_frame], axis=1)

    return result
