from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT, RAW_ATTENDANCE_FILENAME, RAW_EVENT_FILENAME, load_data


def test_load_data_returns_unmodified_raw_columns() -> None:
    booking_events, attendance = load_data(use_synthetic=False)

    assert list(booking_events.columns) == [
        "studio",
        "course",
        "attendance list",
        "current_date",
        "class_date",
        "is_holiday",
        "is_holiday_week",
        "maxnr",
        "waiting list",
        "warteliste_length",
    ]
    assert list(attendance.columns) == [
        "studio",
        "course",
        "attendance list",
        "waiting list",
        "instructor",
        "date",
        "maxnr",
    ]


def test_synthetic_raw_columns_match_production_raw_columns() -> None:
    raw_booking_events, raw_attendance = load_data(use_synthetic=False)
    synthetic_booking_events, synthetic_attendance = load_data(use_synthetic=True)

    assert list(synthetic_booking_events.columns) == list(raw_booking_events.columns)
    assert list(synthetic_attendance.columns) == list(raw_attendance.columns)


def test_load_data_matches_direct_csv_reads() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    booking_events, attendance = load_data(use_synthetic=True)
    direct_booking_events = pd.read_csv(repo_root / "data" / "synthetic" / "event_log.csv")
    direct_attendance = pd.read_csv(repo_root / "data" / "synthetic" / "attendance_log.csv")

    pd.testing.assert_frame_equal(booking_events, direct_booking_events)
    pd.testing.assert_frame_equal(attendance, direct_attendance)


def test_project_root_matches_repository_root() -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_expected_csv_filenames_are_stable() -> None:
    assert RAW_EVENT_FILENAME == "event_log.csv"
    assert RAW_ATTENDANCE_FILENAME == "attendance_log.csv"
