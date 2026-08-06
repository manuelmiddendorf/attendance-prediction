"""Prepare canonical datasets for descriptive data analysis.

This module adds analysis-ready summary columns to the canonical attendance
and booking-event tables created in the data layer.
The functions are intentionally observational: they copy the incoming
DataFrames and derive counts, time offsets, and calendar attributes without
changing the underlying booking states, timestamps, or business values.
That keeps notebook code concise while preserving one reusable implementation
for descriptive analysis steps that may also be reused in later pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _count_member_collection(value: object) -> object:
    """Count members in a canonical list column while preserving missing values.

    Parameters
    ----------
    value : object
        Single cell value from a canonical member-list column.

    Returns
    -------
    int | pd.NA
        Length of the list when the value is present, otherwise ``pd.NA``.

    Raises
    ------
    TypeError
        Raised when the value is neither a list nor a missing value marker.
    """

    if isinstance(value, list):
        return len(value)
    if value is None or pd.isna(value):
        return pd.NA

    raise TypeError(
        "Expected a canonical member collection to be a list or missing value."
    )


def add_attendance_counts(attendance: pd.DataFrame) -> pd.DataFrame:
    """Add observed attendance counts to the canonical attendance table.

    Parameters
    ----------
    attendance : pd.DataFrame
        Canonical attendance table with an ``attendance_list`` column.

    Returns
    -------
    pd.DataFrame
        Copy of ``attendance`` with an added ``attendance_count`` column.
    """

    result = attendance.copy()
    result["attendance_count"] = result["attendance_list"].map(_count_member_collection)
    return result


def add_waiting_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Add observed waiting-list counts to a canonical analysis table.

    Parameters
    ----------
    frame : pd.DataFrame
        Canonical attendance or booking-events table with a ``waiting_list``
        column.

    Returns
    -------
    pd.DataFrame
        Copy of ``frame`` with an added ``waiting_count`` column.
    """

    result = frame.copy()
    result["waiting_count"] = result["waiting_list"].map(_count_member_collection)
    return result


def add_occupancy_rates(attendance: pd.DataFrame) -> pd.DataFrame:
    """Add attendance-based occupancy rates when class capacity is available.

    Parameters
    ----------
    attendance : pd.DataFrame
        Attendance analysis table with ``attendance_count`` and optionally
        ``capacity``.

    Returns
    -------
    pd.DataFrame
        Copy of ``attendance`` with an ``occupancy_rate`` column when
        ``capacity`` exists, otherwise an unchanged copy.
    """

    result = attendance.copy()

    if "capacity" in result.columns:
        result["occupancy_rate"] = result["attendance_count"] / result["capacity"]

    return result


def add_temporal_columns(attendance: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived columns from the class start timestamp.

    Parameters
    ----------
    attendance : pd.DataFrame
        Canonical attendance table with a ``class_start`` timestamp column.

    Returns
    -------
    pd.DataFrame
        Copy of ``attendance`` with added ``weekday``, ``class_hour``, and
        ``class_date`` columns.
    """

    result = attendance.copy()
    result["weekday"] = result["class_start"].dt.day_name()
    result["class_hour"] = result["class_start"].dt.hour
    result["class_date"] = result["class_start"].dt.date
    return result


def prepare_attendance_analysis(attendance: pd.DataFrame) -> pd.DataFrame:
    """Prepare the canonical attendance table for descriptive analysis.

    Parameters
    ----------
    attendance : pd.DataFrame
        Canonical attendance table produced by ``prepare_attendance``.

    Returns
    -------
    pd.DataFrame
        Copy of ``attendance`` with attendance counts, waiting-list counts,
        calendar attributes, and occupancy rates.
    """

    result = add_attendance_counts(attendance)
    result = add_waiting_counts(result)
    result = add_temporal_columns(result)
    result = add_occupancy_rates(result)
    return result


def prepare_booking_events_analysis(booking_events: pd.DataFrame) -> pd.DataFrame:
    """Prepare canonical booking snapshots for descriptive analysis.

    Parameters
    ----------
    booking_events : pd.DataFrame
        Canonical booking-events table produced by ``prepare_booking_events``.

    Returns
    -------
    pd.DataFrame
        Copy of ``booking_events`` with booked-member counts, waiting-list
        counts, and lead times before class start measured in hours and days.
    """

    result = add_waiting_counts(booking_events)
    result["booked_count"] = result["attendance_list"].map(_count_member_collection)
    result["hours_before_class"] = (
        result["class_start"] - result["event_timestamp"]
    ).dt.total_seconds() / 3600
    result["days_before_class"] = result["hours_before_class"] / 24
    return result


def create_booking_trajectory(
    booking_events: pd.DataFrame,
    horizons_hours: Sequence[float | int],
    class_columns: Sequence[str],
) -> pd.DataFrame:
    """Extract the latest known booking state at selected prediction horizons.

    For every class and every requested horizon, the function selects the most
    recent booking snapshot that was available at least that many hours before
    class start.

    Parameters
    ----------
    booking_events : pd.DataFrame
        Booking snapshot data with ``class_start``, ``event_timestamp``,
        ``attendance_list``, and ``waiting_list``.
    horizons_hours : Sequence[float | int]
        Prediction horizons measured in hours before class start.
    class_columns : Sequence[str]
        Columns identifying a unique class.

    Returns
    -------
    pd.DataFrame
        One row per class and prediction horizon, with snapshot columns filled
        from the latest qualifying booking state when such a state exists.
    """

    analysis_frame = prepare_booking_events_analysis(booking_events)
    horizon_column = "horizon_hours"

    class_frame = analysis_frame.loc[:, list(class_columns)].drop_duplicates()
    horizon_frame = pd.DataFrame({horizon_column: list(horizons_hours)})
    class_horizon_grid = class_frame.merge(horizon_frame, how="cross")

    eligible = analysis_frame.merge(horizon_frame, how="cross")
    eligible = eligible.loc[
        eligible["hours_before_class"] >= eligible[horizon_column]
    ].copy()

    if eligible.empty:
        return class_horizon_grid

    latest_snapshots = (
        eligible.sort_values([*class_columns, horizon_column, "event_timestamp"])
        .groupby([*class_columns, horizon_column], as_index=False)
        .tail(1)
    )

    return class_horizon_grid.merge(
        latest_snapshots,
        on=[*class_columns, horizon_column],
        how="left",
    )
