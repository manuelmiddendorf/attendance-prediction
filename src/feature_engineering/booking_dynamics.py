"""Add simple booking-dynamics features from pre-prediction event histories.

This module extends the incremental feature table with leakage-safe summaries
describing how the attendance booking state of the current class evolved before
prediction time.
It uses only booking snapshots that were available at or before the current
prediction instance and treats positive or negative changes in booked-member
counts as observed booking transitions.
The implementation remains intentionally explicit and prioritizes
interpretability over aggressive optimization.
"""

from __future__ import annotations

import pandas as pd


_BOOKING_DYNAMICS_WINDOWS = {
    "6h": pd.Timedelta(hours=6),
    "24h": pd.Timedelta(hours=24),
    "72h": pd.Timedelta(hours=72),
}

_CLASS_COLUMNS = ["studio", "course", "class_start"]


def _build_booking_timelines_by_class(
    booking_events_history: pd.DataFrame,
) -> dict[tuple[object, object, object], pd.DataFrame]:
    """Build sorted per-class booking timelines with attendance deltas.

    Parameters
    ----------
    booking_events_history : pd.DataFrame
        Canonical booking-event history with ``event_timestamp``,
        ``event_order``, and ``attendance_list`` columns.

    Returns
    -------
    dict[tuple[object, object, object], pd.DataFrame]
        Mapping from class key to a booking timeline sorted by
        ``event_timestamp`` and ``event_order``. Each timeline contains
        ``attendance_count`` and ``attendance_delta`` columns.
    """

    sorted_history = booking_events_history.sort_values(
        _CLASS_COLUMNS + ["event_timestamp", "event_order"]
    ).copy()
    sorted_history["attendance_count"] = sorted_history["attendance_list"].map(len)

    timelines_by_class: dict[tuple[object, object, object], pd.DataFrame] = {}

    for class_key, class_history in sorted_history.groupby(_CLASS_COLUMNS):
        timeline = class_history.loc[:, ["event_timestamp", "attendance_count"]].copy()
        timeline["attendance_delta"] = timeline["attendance_count"].diff()
        timelines_by_class[class_key] = timeline

    return timelines_by_class


def _build_booking_dynamics_for_row(
    class_timeline: pd.DataFrame,
    prediction_time: pd.Timestamp,
    current_attendance_count: int,
) -> dict[str, int]:
    """Summarize observed booking dynamics for one prediction instance.

    Parameters
    ----------
    class_timeline : pd.DataFrame
        Sorted booking timeline for one class with ``event_timestamp``,
        ``attendance_count``, and ``attendance_delta`` columns.
    prediction_time : pd.Timestamp
        Time at which the current prediction instance is observed.
    current_attendance_count : int
        Number of members currently booked in the prediction instance.

    Returns
    -------
    dict[str, int]
        Booking-dynamics feature values for the current prediction instance.
    """

    eligible_timeline = class_timeline.loc[
        class_timeline["event_timestamp"] <= prediction_time
    ]

    if eligible_timeline.empty:
        raise ValueError(
            "Prediction instances must have at least one booking snapshot at or "
            "before prediction_time."
        )

    transition_history = eligible_timeline.loc[
        eligible_timeline["attendance_delta"].notna()
    ].copy()

    feature_values: dict[str, int] = {}

    for window_label, window in _BOOKING_DYNAMICS_WINDOWS.items():
        window_start = prediction_time - window
        recent_transitions = transition_history.loc[
            transition_history["event_timestamp"] > window_start
        ]

        feature_values[f"observed_booking_increase_events_{window_label}"] = int(
            (recent_transitions["attendance_delta"] > 0).sum()
        )
        feature_values[f"observed_booking_decrease_events_{window_label}"] = int(
            (recent_transitions["attendance_delta"] < 0).sum()
        )
        feature_values[f"observed_net_booking_change_{window_label}"] = int(
            recent_transitions["attendance_delta"].sum()
        )

    max_observed_attendance = int(eligible_timeline["attendance_count"].max())
    feature_values["max_observed_attendance_before_prediction"] = (
        max_observed_attendance
    )
    feature_values["attendance_drop_from_observed_peak"] = (
        max_observed_attendance - current_attendance_count
    )

    return feature_values


def add_booking_dynamics_features(
    features: pd.DataFrame,
    booking_events_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe booking-dynamics features to the current table.

    Parameters
    ----------
    features : pd.DataFrame
        Current incremental feature table. It must contain ``studio``,
        ``course``, ``class_start``, ``attendance_list``, and
        ``prediction_time`` columns.
    booking_events_history : pd.DataFrame
        Canonical booking-event history with ``event_timestamp`` and
        ``event_order`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with additional booking-dynamics features derived
        from booking snapshots available at or before each prediction time.
    """

    result = features.copy()
    booking_timelines_by_class = _build_booking_timelines_by_class(
        booking_events_history,
    )
    booking_dynamics_rows = []

    for row in result.itertuples(index=False):
        class_key = (row.studio, row.course, row.class_start)
        class_timeline = booking_timelines_by_class.get(class_key)

        if class_timeline is None:
            raise ValueError(
                "Every prediction instance must correspond to booking-event "
                "history for the same class."
            )

        booking_dynamics_rows.append(
            _build_booking_dynamics_for_row(
                class_timeline,
                row.prediction_time,
                len(row.attendance_list),
            )
        )

    booking_dynamics_frame = pd.DataFrame(booking_dynamics_rows, index=result.index)
    return pd.concat([result, booking_dynamics_frame], axis=1)
