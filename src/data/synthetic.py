"""Deterministic synthetic data generation for notebook and test workflows.

This module creates privacy-safe CSV replacements that mimic the structure of
the protected raw exports used by the project.
It preserves the original source headers so loading and normalization can be
tested against realistic input shapes without exposing personal data.
The generated files intentionally include a small number of quality issues,
such as duplicates and missing values, so exploratory checks and validation
logic can be exercised under controlled conditions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import RAW_ATTENDANCE_FILENAME, RAW_EVENT_FILENAME

DEFAULT_SYNTHETIC_SEED = 20260723


@dataclass(frozen=True)
class SyntheticDatasets:
    """Container for the two raw-schema synthetic datasets."""

    event_log: pd.DataFrame
    attendance_log: pd.DataFrame


def _course_schedules() -> list[dict[str, Any]]:
    """Return the recurring class templates used for synthetic generation.

    Parameters
    ----------
    None
        This function returns static schedule metadata.

    Returns
    -------
    list[dict[str, Any]]
        List of recurring course definitions with studio, weekday, capacity,
        instructor, and demand settings.
    """

    return [
        {"studio": "Cb", "course": "Di 11:00", "weekday": 1, "capacity": 12, "instructor": "Marta", "demand": 1.0},
        {"studio": "Cb", "course": "Do 11:00", "weekday": 3, "capacity": 12, "instructor": "Marta", "demand": 0.95},
        {"studio": "Cb", "course": "Fr 17:00", "weekday": 4, "capacity": 12, "instructor": "Manuel", "demand": 1.15},
        {"studio": "Cb", "course": "Mi 16:00", "weekday": 2, "capacity": 14, "instructor": "Marta", "demand": 1.05},
        {"studio": "Nk", "course": "Mo 18:00", "weekday": 0, "capacity": 14, "instructor": "Clara", "demand": 1.1},
        {"studio": "Nk", "course": "Di 18:00", "weekday": 1, "capacity": 12, "instructor": "Jonas", "demand": 0.9},
        {"studio": "Nk", "course": "Mi 20:00", "weekday": 2, "capacity": 10, "instructor": "Nina", "demand": 0.85},
        {"studio": "Nk", "course": "Sa 09:00", "weekday": 5, "capacity": 16, "instructor": "Clara", "demand": 1.0},
    ]


def _holiday_dates() -> set[pd.Timestamp]:
    """Return individual holiday dates injected into the synthetic calendar.

    Parameters
    ----------
    None
        This function returns static holiday dates.

    Returns
    -------
    set[pd.Timestamp]
        Calendar dates that should be marked as holidays in synthetic rows.
    """

    return {
        pd.Timestamp("2026-04-03"),
        pd.Timestamp("2026-05-01"),
    }


def _holiday_weeks() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return holiday-week intervals for the synthetic calendar.

    Parameters
    ----------
    None
        This function returns static date ranges.

    Returns
    -------
    list[tuple[pd.Timestamp, pd.Timestamp]]
        Inclusive start and end timestamps for holiday-week periods.
    """

    return [
        (pd.Timestamp("2026-02-16"), pd.Timestamp("2026-02-22")),
        (pd.Timestamp("2026-03-30"), pd.Timestamp("2026-04-05")),
        (pd.Timestamp("2026-04-27"), pd.Timestamp("2026-05-03")),
    ]


def _is_holiday_week(class_day: pd.Timestamp) -> bool:
    """Check whether a class day falls inside a configured holiday week.

    Parameters
    ----------
    class_day : pd.Timestamp
        Normalized class-day timestamp.

    Returns
    -------
    bool
        ``True`` when the date lies inside any synthetic holiday-week range.
    """

    return any(start <= class_day <= end for start, end in _holiday_weeks())


def _serialize_members(members: list[str] | None) -> str | None:
    """Serialize a member list into the raw CSV string representation.

    Parameters
    ----------
    members : list[str] | None
        Member identifiers to serialize, or ``None`` for a missing value.

    Returns
    -------
    str | None
        Compact JSON string representation, or ``None``.
    """

    if members is None:
        return None
    return json.dumps(members, separators=(",", ":"))


def _serialize_bool(value: Any) -> str | None:
    """Serialize a boolean-like value into the raw CSV representation.

    Parameters
    ----------
    value : Any
        Boolean-like value that may also be missing.

    Returns
    -------
    str | None
        Lowercase string ``"true"`` or ``"false"``, or ``None`` for missing
        values.
    """

    if pd.isna(value):
        return None
    return "true" if bool(value) else "false"


def create_synthetic_raw_datasets(seed: int = DEFAULT_SYNTHETIC_SEED) -> SyntheticDatasets:
    """Create deterministic synthetic datasets using the original raw headers.

    Parameters
    ----------
    seed : int, optional
        Deterministic random seed controlling the generated synthetic data.

    Returns
    -------
    SyntheticDatasets
        Raw-schema synthetic event and attendance DataFrames.
    """

    rng = np.random.default_rng(seed)
    schedules = _course_schedules()
    week_starts = pd.date_range("2026-01-05", periods=22, freq="W-MON")
    holiday_dates = _holiday_dates()

    member_pools = {
        "Cb": [f"cb_member_{idx:03d}" for idx in range(1, 81)],
        "Nk": [f"nk_member_{idx:03d}" for idx in range(1, 81)],
    }

    attendance_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    class_counter = 0

    for schedule in schedules:
        for week_start in week_starts:
            class_day = week_start + pd.Timedelta(days=schedule["weekday"])
            class_start = pd.Timestamp(f"{class_day.date()} {schedule['course'].split()[1]}")
            holiday = class_day.normalize() in holiday_dates
            holiday_week = _is_holiday_week(class_day.normalize())

            demand_signal = (
                schedule["capacity"] * schedule["demand"]
                - (2.0 if holiday else 0.8 if holiday_week else 0.0)
                + rng.normal(0.0, 1.7)
            )
            signup_count = int(np.clip(round(min(schedule["capacity"], demand_signal)), 1, schedule["capacity"]))
            waiting_count = int(np.clip(round(max(demand_signal - schedule["capacity"], 0) + rng.normal(0.3, 0.9)), 0, 4))

            sample = rng.choice(member_pools[schedule["studio"]], size=signup_count + waiting_count + 3, replace=False)
            attendees = sample[:signup_count].tolist()
            waiting_list = sample[signup_count : signup_count + waiting_count].tolist()
            transient = sample[signup_count + waiting_count :].tolist()

            attendance_rows.append(
                {
                    "studio": schedule["studio"],
                    "course": schedule["course"],
                    "attendance list": _serialize_members(attendees.copy()),
                    "waiting list": _serialize_members(waiting_list.copy()),
                    "instructor": schedule["instructor"],
                    "date": (
                        class_start.tz_localize("Europe/Berlin")
                        .tz_convert("UTC")
                        .strftime("%Y-%m-%dT%H:%M:%S.000Z")
                    ),
                    "maxnr": schedule["capacity"],
                }
            )

            base_offsets = [168, 132, 96, 72, 48, 24, 12, 6]
            offset_count = int(rng.integers(5, 8))
            offsets = sorted(rng.choice(base_offsets, size=offset_count, replace=False).tolist(), reverse=True)
            if class_counter % 18 == 0:
                offsets.append(0)
            if class_counter % 25 == 0:
                offsets.append(-2)
            offsets = sorted(offsets, reverse=True)

            for index, offset_hours in enumerate(offsets):
                progress = index / max(len(offsets) - 1, 1)
                signup_target = int(np.clip(round(signup_count * (0.35 + 0.7 * progress)), 1, signup_count))
                waiting_target = int(np.clip(round(waiting_count * max(progress - 0.35, 0) / 0.65), 0, waiting_count))

                snapshot_attendance = attendees[:signup_target].copy()
                snapshot_waiting = waiting_list[:waiting_target].copy()
                if progress < 0.45 and transient:
                    snapshot_attendance.append(transient[0])
                if progress < 0.3 and len(transient) > 1:
                    snapshot_waiting.append(transient[1])
                if class_counter % 14 == 0 and index == len(offsets) - 1 and len(snapshot_attendance) > 1:
                    snapshot_attendance = snapshot_attendance[:-1]
                if class_counter % 17 == 0 and index == len(offsets) - 1 and waiting_list:
                    snapshot_waiting = waiting_list[:1]

                snapshot_rows.append(
                    {
                        "studio": schedule["studio"],
                        "course": schedule["course"],
                        "attendance list": _serialize_members(snapshot_attendance),
                        "current_date": (class_start - pd.Timedelta(hours=offset_hours)).strftime("%Y-%m-%d %H:%M:%S"),
                        "class_date": class_day.normalize().strftime("%Y-%m-%d"),
                        "is_holiday": _serialize_bool(holiday),
                        "is_holiday_week": _serialize_bool(holiday_week),
                        "maxnr": schedule["capacity"],
                        "waiting list": _serialize_members(snapshot_waiting),
                        "warteliste_length": len(snapshot_waiting),
                    }
                )

            class_counter += 1

    attendance_log = pd.DataFrame(attendance_rows).sort_values(["studio", "course", "date"]).reset_index(drop=True)
    event_log = pd.DataFrame(snapshot_rows).sort_values(["studio", "course", "class_date", "current_date"]).reset_index(drop=True)

    event_only_mask = (
        (event_log["studio"] == "Nk")
        & (event_log["course"] == "Mi 20:00")
        & (event_log["class_date"] == "2026-03-18")
    )
    attendance_only_mask = (
        (attendance_log["studio"] == "Cb")
        & (attendance_log["course"] == "Do 11:00")
        & (attendance_log["date"] == "2026-04-09T09:00:00.000Z")
    )

    event_log = event_log.loc[~event_only_mask].reset_index(drop=True)
    attendance_log = attendance_log.loc[~attendance_only_mask].reset_index(drop=True)

    duplicate_snapshot = event_log.iloc[[12]].copy()
    event_log = pd.concat([event_log, duplicate_snapshot], ignore_index=True)

    attendance_log.loc[5, "instructor"] = "MARTA"
    attendance_log.loc[16, "instructor"] = "Martha"
    attendance_log.loc[23, "instructor"] = None
    attendance_log.loc[31, "maxnr"] = None

    event_log.loc[9, "waiting list"] = None
    event_log.loc[17, "is_holiday_week"] = None
    event_log.loc[24, "maxnr"] = None

    attendance_members = json.loads(attendance_log.at[8, "attendance list"])
    if len(attendance_members) >= 2:
        attendance_log.at[8, "attendance list"] = _serialize_members(attendance_members + [attendance_members[0]])
        attendance_log.at[8, "waiting list"] = _serialize_members([attendance_members[1], attendance_members[1]])

    snapshot_members = json.loads(event_log.at[3, "attendance list"])
    if len(snapshot_members) >= 2:
        event_log.at[3, "attendance list"] = _serialize_members(snapshot_members + [snapshot_members[0]])
        event_log.at[3, "waiting list"] = _serialize_members([snapshot_members[1]])
        event_log.at[3, "warteliste_length"] = 1

    return SyntheticDatasets(event_log=event_log, attendance_log=attendance_log)


def ensure_synthetic_data_files(
    repo_root: Path,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write deterministic synthetic CSVs to ``data/synthetic`` when needed.

    Parameters
    ----------
    repo_root : Path
        Repository root containing the ``data`` directory.
    seed : int, optional
        Deterministic random seed for synthetic generation.
    overwrite : bool, optional
        If ``True``, regenerate files even when they already exist.

    Returns
    -------
    dict[str, Path]
        Paths to the generated or reused synthetic event and attendance files.
    """

    data_dir = repo_root / "data" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    event_path = data_dir / RAW_EVENT_FILENAME
    attendance_path = data_dir / RAW_ATTENDANCE_FILENAME

    if overwrite or not event_path.exists() or not attendance_path.exists():
        datasets = create_synthetic_raw_datasets(seed=seed)
        datasets.event_log.to_csv(event_path, index=False)
        datasets.attendance_log.to_csv(attendance_path, index=False)

    return {"event_log": event_path, "attendance_log": attendance_path}


def create_synthetic_data(seed: int = DEFAULT_SYNTHETIC_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the two raw-schema synthetic DataFrames for backward compatibility.

    Parameters
    ----------
    seed : int, optional
        Deterministic random seed for synthetic generation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Synthetic event and attendance DataFrames using the original raw CSV
        schema.
    """

    datasets = create_synthetic_raw_datasets(seed=seed)
    return datasets.event_log.copy(), datasets.attendance_log.copy()
