"""Add simple member-level historical attendance features.

This module extends the incremental feature table with summaries describing
the historical attendance behavior of members who are currently booked at
prediction time.
For each prediction instance it looks only at classes that started before the
prediction time and aggregates per-member attendance counts and affinities for
the current course and instructor across simple historical windows.
The implementation is intentionally explicit and favors readability over
optimization because the current project dataset is small.
"""

from __future__ import annotations

import pandas as pd


_MEMBER_HISTORY_WINDOWS = {
    "30d": pd.Timedelta(days=30),
    "90d": pd.Timedelta(days=90),
    "365d": pd.Timedelta(days=365),
}


def _select_window_history(
    attendance_history: pd.DataFrame,
    prediction_time: pd.Timestamp,
    window: pd.Timedelta,
) -> pd.DataFrame:
    """Select historical classes in one window before a prediction time.

    Parameters
    ----------
    attendance_history : pd.DataFrame
        Clean canonical attendance history with ``class_start`` and
        ``attendance_list`` columns.
    prediction_time : pd.Timestamp
        Time at which the prediction instance is observed.
    window : pd.Timedelta
        Historical lookback window.

    Returns
    -------
    pd.DataFrame
        Historical attendance rows satisfying
        ``prediction_time - window <= class_start < prediction_time``.
    """

    window_start = prediction_time - window
    return attendance_history.loc[
        (attendance_history["class_start"] >= window_start)
        & (attendance_history["class_start"] < prediction_time)
    ]


def _build_window_member_features(
    window_history: pd.DataFrame,
    current_members: list[str],
    current_course: str,
    current_instructor: str,
) -> dict[str, object]:
    """Aggregate member attendance behavior for one historical window.

    Parameters
    ----------
    window_history : pd.DataFrame
        Historical attendance rows already restricted to one prediction window.
    current_members : list[str]
        Member IDs currently contained in the prediction instance's
        ``attendance_list``.
    current_course : str
        Course label of the prediction instance.
    current_instructor : str
        Instructor label of the prediction instance.

    Returns
    -------
    dict[str, object]
        Aggregated member-level quantities for the selected historical window.
    """

    member_attendance_counts: list[int] = []
    member_course_affinities: list[float] = []
    member_instructor_affinities: list[float] = []

    for member_id in current_members:
        member_history = window_history.loc[
            window_history["attendance_list"].map(lambda members: member_id in members)
        ]

        attendance_count = len(member_history)
        member_attendance_counts.append(attendance_count)

        if attendance_count > 0:
            member_course_affinities.append(
                (member_history["course"] == current_course).mean()
            )
            member_instructor_affinities.append(
                (member_history["instructor"] == current_instructor).mean()
            )

    attendance_count_mean = pd.Series(member_attendance_counts, dtype="float64").mean()
    members_with_history_count = sum(count > 0 for count in member_attendance_counts)
    members_with_history_share = (
        members_with_history_count / len(current_members)
        if len(current_members) > 0
        else float("nan")
    )

    return {
        "member_attendance_count_mean": attendance_count_mean,
        "members_with_history_count": members_with_history_count,
        "members_with_history_share": members_with_history_share,
        "member_course_affinity_mean": pd.Series(
            member_course_affinities,
            dtype="float64",
        ).mean(),
        "member_instructor_affinity_mean": pd.Series(
            member_instructor_affinities,
            dtype="float64",
        ).mean(),
    }


def add_member_history_features(
    features: pd.DataFrame,
    attendance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe member-level historical features to the current table.

    Parameters
    ----------
    features : pd.DataFrame
        Current incremental feature table. It must contain ``attendance_list``,
        ``course``, ``instructor``, and ``prediction_time`` columns.
    attendance_history : pd.DataFrame
        Clean canonical attendance history. Only classes before each
        prediction time are allowed to contribute to member-history features.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with additional member-level attendance-history
        and affinity features for the 30-day, 90-day, and 365-day windows.
    """

    result = features.copy()
    member_feature_rows = []

    for row in result.itertuples(index=False):
        row_features: dict[str, object] = {}

        for window_label, window in _MEMBER_HISTORY_WINDOWS.items():
            window_history = _select_window_history(
                attendance_history,
                row.prediction_time,
                window,
            )
            window_features = _build_window_member_features(
                window_history,
                row.attendance_list,
                row.course,
                row.instructor,
            )

            for feature_name, value in window_features.items():
                row_features[f"{feature_name}_{window_label}"] = value

        member_feature_rows.append(row_features)

    member_feature_frame = pd.DataFrame(member_feature_rows, index=result.index)
    return pd.concat([result, member_feature_frame], axis=1)
