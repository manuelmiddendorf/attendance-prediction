from __future__ import annotations

import csv
from pathlib import Path

from src.data.synthetic import DEFAULT_SYNTHETIC_SEED, create_synthetic_raw_datasets, ensure_synthetic_data_files


def test_synthetic_generation_is_deterministic() -> None:
    first = create_synthetic_raw_datasets(seed=DEFAULT_SYNTHETIC_SEED)
    second = create_synthetic_raw_datasets(seed=DEFAULT_SYNTHETIC_SEED)

    assert first.event_log.equals(second.event_log)
    assert first.attendance_log.equals(second.attendance_log)


def test_synthetic_headers_match_raw_headers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ensure_synthetic_data_files(repo_root, overwrite=True)

    pairs = [
        ("data/raw/event_log.csv", "data/synthetic/event_log.csv"),
        ("data/raw/attendance_log.csv", "data/synthetic/attendance_log.csv"),
    ]

    for raw_rel, synthetic_rel in pairs:
        raw_path = repo_root / raw_rel
        synthetic_path = repo_root / synthetic_rel

        with raw_path.open(newline="", encoding="utf-8") as raw_handle:
            raw_header = next(csv.reader(raw_handle))
        with synthetic_path.open(newline="", encoding="utf-8") as synthetic_handle:
            synthetic_header = next(csv.reader(synthetic_handle))

        assert raw_header == synthetic_header
