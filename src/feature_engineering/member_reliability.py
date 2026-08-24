"""Add member reliability features based on historical booking states. """

from __future__ import annotations

import pandas as pd


_MEMBER_RELIABILITY_WINDOWS = {
    "30d": pd.Timedelta(days=30),
    "90d": pd.Timedelta(days=90),
    "365d": pd.Timedelta(days=365),
}

_CLASS_COLUMNS = ["studio", "course", "class_start"]


def _select_window_attendance_history(
    attendance_history: pd.DataFrame,
    prediction_time: pd.Timestamp,
    window: pd.Timedelta,
) -> pd.DataFrame:
    """Select historical classes in one window before a prediction time.

    Parameters
    ----------
    attendance_history : pd.DataFrame
        Clean canonical attendance history.
    prediction_time : pd.Timestamp
        Time at which the current prediction instance is observed.
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


def _build_booking_events_by_class(
    booking_events_history: pd.DataFrame,
    attendance_history: pd.DataFrame,
) -> dict[tuple[object, object, object], pd.DataFrame]:
    """Group booking-event histories by class after attendance-based filtering.

    Parameters
    ----------
    booking_events_history : pd.DataFrame
        Canonical booking-event history with an ``event_order`` column.
    attendance_history : pd.DataFrame
        Clean canonical attendance history defining the set of classes with
        valid final attendance outcomes.

    Returns
    -------
    dict[tuple[object, object, object], pd.DataFrame]
        Mapping from class key to booking-event history sorted by
        ``event_timestamp`` and ``event_order``.
    """

    valid_class_keys = attendance_history.loc[:, _CLASS_COLUMNS].drop_duplicates()
    filtered_booking_events = booking_events_history.merge(
        valid_class_keys,
        on=_CLASS_COLUMNS,
        how="inner",
    )

    booking_events_by_class: dict[tuple[object, object, object], pd.DataFrame] = {}

    for class_key, class_events in filtered_booking_events.groupby(_CLASS_COLUMNS):
        booking_events_by_class[class_key] = class_events.sort_values(
            ["event_timestamp", "event_order"]
        )

    return booking_events_by_class


def _select_latest_snapshot_at_horizon(
    class_booking_events: pd.DataFrame,
    historical_prediction_time: pd.Timestamp,
) -> pd.Series | None:
    """Select the latest historical booking snapshot available at a horizon.

    Parameters
    ----------
    class_booking_events : pd.DataFrame
        Booking-event history for one class, sorted by ``event_timestamp`` and
        ``event_order``.
    historical_prediction_time : pd.Timestamp
        Prediction time to reconstruct for the historical class.

    Returns
    -------
    pd.Series | None
        Latest eligible snapshot for the historical prediction time, or
        ``None`` when no snapshot existed at or before that time.
    """

    eligible_snapshots = class_booking_events.loc[
        class_booking_events["event_timestamp"] <= historical_prediction_time
    ]

    if eligible_snapshots.empty:
        return None

    return eligible_snapshots.iloc[-1]


def _precompute_historical_snapshots_by_horizon(
    booking_events_history: pd.DataFrame,
    attendance_history: pd.DataFrame,
    prediction_horizons: list[object],
) -> dict[object, dict[tuple[object, object, object], frozenset[str]]]:
    """Precompute latest eligible booking snapshots for each class and horizon.

    Parameters
    ----------
    booking_events_history : pd.DataFrame
        Canonical booking-event history with an ``event_order`` column.
    attendance_history : pd.DataFrame
        Clean canonical attendance history defining the historical classes
        whose final outcomes may be used.
    prediction_horizons : list[object]
        Unique prediction horizons present in the current feature table.

    Returns
    -------
    dict[object, dict[tuple[object, object, object], frozenset[str]]]
        Mapping from prediction horizon to a lookup of historical class keys
        and their latest eligible booked-member snapshots.
    """

    booking_events_by_class = _build_booking_events_by_class(
        booking_events_history,
        attendance_history,
    )
    snapshots_by_horizon: dict[
        object,
        dict[tuple[object, object, object], frozenset[str]],
    ] = {}

    for prediction_horizon in prediction_horizons:
        horizon_snapshots: dict[tuple[object, object, object], frozenset[str]] = {}
        horizon_delta = pd.Timedelta(hours=prediction_horizon)

        for historical_class in attendance_history.itertuples(index=False):
            class_key = (
                historical_class.studio,
                historical_class.course,
                historical_class.class_start,
            )
            class_booking_events = booking_events_by_class.get(class_key)

            if class_booking_events is None:
                continue

            historical_prediction_time = historical_class.class_start - horizon_delta
            historical_snapshot = _select_latest_snapshot_at_horizon(
                class_booking_events,
                historical_prediction_time,
            )

            if historical_snapshot is None:
                continue

            horizon_snapshots[class_key] = frozenset(
                historical_snapshot["attendance_list"]
            )

        snapshots_by_horizon[prediction_horizon] = horizon_snapshots

    return snapshots_by_horizon


def _build_window_member_reliability_features(
    window_attendance_history: pd.DataFrame,
    historical_snapshots: dict[tuple[object, object, object], frozenset[str]],
    current_members: list[str],
) -> dict[str, object]:
    """Aggregate member reliability statistics for one historical window.

    Parameters
    ----------
    window_attendance_history : pd.DataFrame
        Historical attendance rows already restricted to one prediction window.
        The rows must include an ``attendance_set`` column for repeated
        membership checks.
    historical_snapshots : dict[tuple[object, object, object], frozenset[str]]
        Latest eligible booked-member snapshots for one prediction horizon.
    current_members : list[str]
        Member IDs currently contained in the prediction instance's
        ``attendance_list``.

    Returns
    -------
    dict[str, object]
        Aggregated reliability quantities for the selected historical window.
    """

    booked_at_horizon_counts = {member_id: 0 for member_id in current_members}
    attended_after_booking_counts = {member_id: 0 for member_id in current_members}

    for historical_class in window_attendance_history.itertuples(index=False):
        class_key = (
            historical_class.studio,
            historical_class.course,
            historical_class.class_start,
        )
        booked_members = historical_snapshots.get(class_key)

        if booked_members is None:
            continue

        attending_members = historical_class.attendance_set

        for member_id in current_members:
            if member_id not in booked_members:
                continue

            booked_at_horizon_counts[member_id] += 1

            if member_id in attending_members:
                attended_after_booking_counts[member_id] += 1

    member_booked_counts = list(booked_at_horizon_counts.values())
    members_with_reliability_history_count = sum(
        count > 0 for count in member_booked_counts
    )
    member_show_up_rates = [
        attended_after_booking_counts[member_id] / booked_at_horizon_counts[member_id]
        for member_id in current_members
        if booked_at_horizon_counts[member_id] > 0
    ]

    return {
        "member_booked_at_horizon_count_mean": pd.Series(
            member_booked_counts,
            dtype="float64",
        ).mean(),
        "members_with_reliability_history_count": (
            members_with_reliability_history_count
        ),
        "members_with_reliability_history_share": (
            members_with_reliability_history_count / len(current_members)
            if len(current_members) > 0
            else float("nan")
        ),
        "member_show_up_rate_mean": pd.Series(
            member_show_up_rates,
            dtype="float64",
        ).mean(),
    }


def add_member_reliability_features(
    features: pd.DataFrame,
    booking_events_history: pd.DataFrame,
    attendance_history: pd.DataFrame,
) -> pd.DataFrame:
    """Add leakage-safe member reliability features to the current table.

    Parameters
    ----------
    features : pd.DataFrame
        Current incremental feature table. It must contain ``attendance_list``,
        ``prediction_time``, and ``prediction_horizon`` columns.
    booking_events_history : pd.DataFrame
        Canonical booking-event history with an ``event_order`` column. The
        latest snapshot is selected using ``event_timestamp`` and
        ``event_order``.
    attendance_history : pd.DataFrame
        Clean canonical attendance history. Only classes before each
        prediction time are allowed to contribute to reliability features.

    Returns
    -------
    pd.DataFrame
        Copy of ``features`` with additional member reliability features for
        the 30-day, 90-day, and 365-day windows.
    """

    result = features.copy()
    historical_attendance = attendance_history.copy()
    historical_attendance["attendance_set"] = historical_attendance[
        "attendance_list"
    ].map(frozenset)
    prediction_horizons = result["prediction_horizon"].drop_duplicates().tolist()
    historical_snapshots_by_horizon = _precompute_historical_snapshots_by_horizon(
        booking_events_history,
        historical_attendance,
        prediction_horizons,
    )
    reliability_feature_rows = []

    for row in result.itertuples(index=False):
        row_features: dict[str, object] = {}

        for window_label, window in _MEMBER_RELIABILITY_WINDOWS.items():
            window_attendance_history = _select_window_attendance_history(
                historical_attendance,
                row.prediction_time,
                window,
            )
            window_features = _build_window_member_reliability_features(
                window_attendance_history,
                historical_snapshots_by_horizon[row.prediction_horizon],
                row.attendance_list,
            )

            for feature_name, value in window_features.items():
                row_features[f"{feature_name}_{window_label}"] = value

        reliability_feature_rows.append(row_features)

    reliability_feature_frame = pd.DataFrame(
        reliability_feature_rows,
        index=result.index,
    )
    return pd.concat([result, reliability_feature_frame], axis=1)
