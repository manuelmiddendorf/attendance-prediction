"""Create canonical representations of the raw datasets.

This module parses structured fields, assigns canonical column names and data
types, and derives class timestamps without cleaning or removing information.
"""

from __future__ import annotations

import json
import re

import pandas as pd


_BOOKING_EVENT_RENAME_MAP = {
    "attendance list": "attendance_list",
    "current_date": "event_timestamp",
    "maxnr": "capacity",
    "waiting list": "waiting_list",
    "warteliste_length": "waiting_list_length",
}

_ATTENDANCE_RENAME_MAP = {
    "attendance list": "attendance_list",
    "waiting list": "waiting_list",
    "date": "class_start",
    "maxnr": "capacity",
}

_STUDIO_TIMEZONE = "Europe/Berlin"
_COURSE_TIME_PATTERN = re.compile(
    r"(?P<weekday>\S+)\s+(?P<hour>\d{1,2}):(?P<minute>\d{2})"
)

def _parse_member_collection(value: object) -> object:
    """Parse a JSON-encoded member collection while preserving missing values."""

    if pd.isna(value):
        return value

    parsed = json.loads(value)

    if not isinstance(parsed, list):
        raise TypeError("Expected a JSON array.")

    return parsed


def _extract_course_time(course_name: str) -> tuple[int, int]:
    """Extract the hour and minute encoded in a course label."""

    match = _COURSE_TIME_PATTERN.fullmatch(course_name)

    if match is None:
        raise ValueError(f"Could not parse course time from {course_name!r}")

    return int(match["hour"]), int(match["minute"])


def _build_class_start(course: str, class_date: object) -> pd.Timestamp:
    """Combine a class date with the time encoded in the course label."""

    hour, minute = _extract_course_time(course)
    return pd.Timestamp(class_date) + pd.Timedelta(hours=hour, minutes=minute)


def _parse_nullable_boolean(value: Any) -> bool | None:
    """Parse a nullable boolean value from raw CSV content.  """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise ValueError(f"Could not parse boolean value: {value!r}")


def prepare_booking_events(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical booking-events representation.

    Parameters
    ----------
    raw_frame : pd.DataFrame
        Raw booking-events DataFrame loaded directly from ``event_log.csv``.

    Returns
    -------
    pd.DataFrame
        Canonical booking-events DataFrame with normalized column names,
        parsed JSON list columns, parsed timestamps, nullable boolean fields,
        and a derived ``class_start`` timestamp.
    """

    frame = raw_frame.rename(columns=_BOOKING_EVENT_RENAME_MAP).copy()
    frame["attendance_list"] = frame["attendance_list"].map(_parse_member_collection)
    frame["waiting_list"] = frame["waiting_list"].map(_parse_member_collection)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], errors="raise")
    frame["class_date"] = pd.to_datetime(frame["class_date"], errors="raise")
    frame["is_holiday"] = frame["is_holiday"].astype("boolean")
    frame["is_holiday_week"] = frame["is_holiday_week"].astype("boolean")
    frame["capacity"] = pd.to_numeric(frame["capacity"], errors="raise").astype("Int64")
    frame["waiting_list_length"] = pd.to_numeric(
        frame["waiting_list_length"],
        errors="raise",
    ).astype("Int64")
    frame["class_start"] = frame.apply(
        lambda row: _build_class_start(row["course"], row["class_date"]),
        axis=1,
    )
    return frame


def prepare_attendance(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical attendance representation.

    Parameters
    ----------
    raw_frame : pd.DataFrame
        Raw attendance DataFrame loaded directly from ``attendance_log.csv``.

    Returns
    -------
    pd.DataFrame
        Canonical attendance DataFrame with normalized column names, parsed
        JSON list columns, a localized ``class_start`` timestamp, and a
        derived ``class_date`` column.
    """

    frame = raw_frame.rename(columns=_ATTENDANCE_RENAME_MAP).copy()
    frame["attendance_list"] = frame["attendance_list"].map(_parse_member_collection)
    frame["waiting_list"] = frame["waiting_list"].map(_parse_member_collection)
    frame["class_start"] = (
        pd.to_datetime(frame["class_start"], utc=True, errors="coerce")
        .dt.tz_convert(_STUDIO_TIMEZONE)
        .dt.tz_localize(None)
    )
    frame["capacity"] = pd.to_numeric(frame["capacity"], errors="raise").astype("Int64")
    frame["class_date"] = frame["class_start"].dt.normalize()
    return frame
