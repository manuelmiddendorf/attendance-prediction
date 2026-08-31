from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    DEFAULT_SYNTHETIC_SEED,
    SyntheticConfig,
    build_calibration_summary,
    create_synthetic_raw_datasets,
    ensure_synthetic_data_files,
    load_data,
    prepare_attendance,
    prepare_booking_events,
    run_privacy_sanity_checks,
)

BOOKING_RAW_COLUMNS = [
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
ATTENDANCE_RAW_COLUMNS = [
    "studio",
    "course",
    "attendance list",
    "waiting list",
    "instructor",
    "date",
    "maxnr",
]
CLASS_COLUMNS = ["studio", "course", "class_start"]


@pytest.fixture(scope="module")
def public_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load committed public synthetic data once for contract tests."""

    return load_data(use_synthetic=True)


def _small_test_config() -> SyntheticConfig:
    """Return a compact configuration for repeated generator tests."""

    return SyntheticConfig(
        attendance_start="2024-01-01",
        attendance_end="2026-06-30",
        booking_event_start="2025-05-05",
        member_count=180,
        recent_member_count=100,
        event_coverage_probability=1.0,
        class_skip_probability=0.0,
    )


def test_synthetic_generation_is_deterministic() -> None:
    config = _small_test_config()

    first = create_synthetic_raw_datasets(
        seed=DEFAULT_SYNTHETIC_SEED,
        config=config,
    )
    second = create_synthetic_raw_datasets(
        seed=DEFAULT_SYNTHETIC_SEED,
        config=config,
    )

    pd.testing.assert_frame_equal(first.event_log, second.event_log)
    pd.testing.assert_frame_equal(first.attendance_log, second.attendance_log)


def test_synthetic_raw_and_canonical_schemas(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    booking_raw, attendance_raw = public_synthetic_data

    assert booking_raw.columns.tolist() == BOOKING_RAW_COLUMNS
    assert attendance_raw.columns.tolist() == ATTENDANCE_RAW_COLUMNS

    booking_events = prepare_booking_events(booking_raw)
    attendance = prepare_attendance(attendance_raw)

    assert booking_events.columns.tolist() == [
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
    assert attendance.columns.tolist() == [
        "studio",
        "course",
        "attendance_list",
        "waiting_list",
        "instructor",
        "class_start",
        "capacity",
        "class_date",
    ]


def test_synthetic_class_keys_and_snapshots_are_valid(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    booking_raw, attendance_raw = public_synthetic_data
    booking_events = prepare_booking_events(booking_raw)
    attendance = prepare_attendance(attendance_raw)
    booking_events["event_order"] = range(len(booking_events))

    assert not attendance.duplicated(CLASS_COLUMNS).any()
    assert (booking_events["event_timestamp"] <= booking_events["class_start"]).all()
    assert booking_events["waiting_list_length"].equals(
        booking_events["waiting_list"].map(len).astype("Int64")
    )

    for _, timeline in booking_events.groupby(CLASS_COLUMNS):
        assert timeline["event_timestamp"].is_monotonic_increasing
        for _, simultaneous_events in timeline.groupby("event_timestamp"):
            assert simultaneous_events["event_order"].is_monotonic_increasing


def test_synthetic_entities_use_fictional_namespaces(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    booking_raw, attendance_raw = public_synthetic_data
    booking_events = prepare_booking_events(booking_raw)
    attendance = prepare_attendance(attendance_raw)

    assert set(attendance["studio"]) == {"Studio A", "Studio B"}
    assert attendance["instructor"].str.startswith("Instructor ").all()
    assert all(
        member_id.startswith("member_")
        for members in attendance["attendance_list"]
        for member_id in members
    )
    assert all(
        waiting_member["ID"].startswith("member_")
        and waiting_member["email"].endswith("@synthetic.invalid")
        for waiting_members in booking_events["waiting_list"]
        for waiting_member in waiting_members
    )


def test_synthetic_history_supports_reliability_features(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    booking_raw, attendance_raw = public_synthetic_data
    booking_events = prepare_booking_events(booking_raw)
    attendance = prepare_attendance(attendance_raw)
    booking_events["event_order"] = range(len(booking_events))

    matched = attendance[[*CLASS_COLUMNS, "attendance_list"]].merge(
        booking_events,
        on=CLASS_COLUMNS,
        how="inner",
    )
    eligible = matched.loc[
        matched["event_timestamp"]
        <= matched["class_start"] - pd.Timedelta(hours=24)
    ]
    prediction_instances = (
        eligible.sort_values(["event_timestamp", "event_order"])
        .groupby(CLASS_COLUMNS)
        .tail(1)
    )
    booking_history_count = pd.Series(
        [
            member_id
            for members in prediction_instances["attendance_list_y"]
            for member_id in set(members)
        ]
    ).value_counts()

    assert len(prediction_instances) >= 700
    assert (booking_history_count >= 5).sum() >= 100
    assert prediction_instances.loc[
        prediction_instances["class_start"] < pd.Timestamp("2026-02-01")
    ].shape[0] > 400
    assert prediction_instances.loc[
        prediction_instances["class_start"] >= pd.Timestamp("2026-05-01")
    ].shape[0] > 100


def test_calibration_summary_contains_only_aggregate_sections(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    summary = build_calibration_summary(*public_synthetic_data)

    assert set(summary) == {
        "scale",
        "attendance",
        "booking_events",
        "prediction_24h",
        "members",
    }
    assert "member_0001" not in str(summary)
    assert "Studio A" not in str(summary)
    assert "Instructor A" not in str(summary)


def test_generator_writes_only_two_data_files(tmp_path: Path) -> None:
    paths = ensure_synthetic_data_files(
        tmp_path,
        seed=DEFAULT_SYNTHETIC_SEED,
        overwrite=True,
        config=_small_test_config(),
    )

    expected_filenames = {"event_log.csv", "attendance_log.csv"}
    assert {path.name for path in paths.values()} == expected_filenames
    assert {
        path.name for path in (tmp_path / "data" / "synthetic").iterdir()
    } == expected_filenames


def test_privacy_checks_omit_generation_files_without_paths(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    booking_raw, attendance_raw = public_synthetic_data

    checks = run_privacy_sanity_checks(
        booking_raw,
        attendance_raw,
        booking_raw,
        attendance_raw,
    )

    assert "only_expected_generation_files_written" not in checks


def test_privacy_checks_accept_expected_generation_files(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    booking_raw, attendance_raw = public_synthetic_data
    generated_paths = {
        "event_log": tmp_path / "event_log.csv",
        "attendance_log": tmp_path / "attendance_log.csv",
    }

    checks = run_privacy_sanity_checks(
        booking_raw,
        attendance_raw,
        booking_raw,
        attendance_raw,
        generated_paths=generated_paths,
    )

    assert checks["only_expected_generation_files_written"] is True


def test_privacy_checks_reject_unexpected_generation_filename(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
    tmp_path: Path,
) -> None:
    booking_raw, attendance_raw = public_synthetic_data
    generated_paths = {
        "event_log": tmp_path / "event_log.csv",
        "unexpected": tmp_path / "member_mapping.csv",
    }

    checks = run_privacy_sanity_checks(
        booking_raw,
        attendance_raw,
        booking_raw,
        attendance_raw,
        generated_paths=generated_paths,
    )

    assert checks["only_expected_generation_files_written"] is False


def test_local_private_data_has_no_direct_synthetic_overlap(
    public_synthetic_data: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "data" / "raw" / "event_log.csv").exists():
        pytest.skip("Private local data are not available.")

    original_booking, original_attendance = load_data(use_synthetic=False)
    synthetic_booking, synthetic_attendance = public_synthetic_data
    checks = run_privacy_sanity_checks(
        original_booking,
        original_attendance,
        synthetic_booking,
        synthetic_attendance,
    )

    diagnostic_keys = {
        "class_summary_exact_overlap_share",
        "member_summary_exact_overlap_share",
    }
    strict_checks = {
        key: value for key, value in checks.items() if key not in diagnostic_keys
    }
    assert all(strict_checks.values())
    assert checks["class_summary_exact_overlap_share"] < 0.8
    assert checks["member_summary_exact_overlap_share"] < 0.8
