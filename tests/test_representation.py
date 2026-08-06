from __future__ import annotations

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from src.data import load_data, prepare_attendance, prepare_booking_events


def test_prepare_booking_events_builds_canonical_representation() -> None:
    booking_events_raw, _ = load_data(use_synthetic=True)

    booking_events = prepare_booking_events(booking_events_raw)

    assert len(booking_events) == len(booking_events_raw)
    assert list(booking_events.columns) == [
        "studio",
        "course",
        "attendance_list",
        "event_timestamp",
        "class_date",
        "is_holiday",
        "is_holiday_week",
        "capacity",
        "waiting_list",
        "waiting_list_length",
        "class_start",
    ]
    assert isinstance(booking_events.at[0, "attendance_list"], list)
    assert isinstance(booking_events.at[0, "waiting_list"], list)
    assert is_datetime64_any_dtype(booking_events["event_timestamp"])
    assert is_datetime64_any_dtype(booking_events["class_date"])
    assert is_datetime64_any_dtype(booking_events["class_start"])
    assert str(booking_events["capacity"].dtype) == "Int64"
    assert str(booking_events["waiting_list_length"].dtype) == "Int64"
    assert booking_events.at[0, "class_start"] == pd.Timestamp("2026-01-06 11:00:00")


def test_prepare_attendance_builds_canonical_representation() -> None:
    _, attendance_raw = load_data(use_synthetic=True)

    attendance = prepare_attendance(attendance_raw)

    assert len(attendance) == len(attendance_raw)
    assert list(attendance.columns) == [
        "studio",
        "course",
        "attendance_list",
        "waiting_list",
        "instructor",
        "class_start",
        "capacity",
        "class_date",
    ]
    assert isinstance(attendance.at[0, "attendance_list"], list)
    assert isinstance(attendance.at[0, "waiting_list"], list)
    assert is_datetime64_any_dtype(attendance["class_start"])
    assert is_datetime64_any_dtype(attendance["class_date"])
    assert str(attendance["capacity"].dtype) == "Int64"
    assert attendance.at[0, "class_start"] == pd.Timestamp("2026-01-06 11:00:00")
    assert attendance.at[0, "class_date"] == pd.Timestamp("2026-01-06")
