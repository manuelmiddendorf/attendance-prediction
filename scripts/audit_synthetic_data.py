"""Print aggregate synthetic calibration and optional local privacy checks.

This development script uses the repository data layer and never prints rows,
entity labels, member identifiers, or exact private timestamps. By default it
reports only the committed public synthetic data. The optional private mode
adds a rounded original-vs-synthetic aggregate comparison and direct-overlap
sanity checks when ignored local raw files are available. Nothing is written
to disk, and the output must not be interpreted as a formal privacy proof.
"""

from __future__ import annotations

import argparse
import json

from src.data import (
    build_calibration_summary,
    load_data,
    run_privacy_sanity_checks,
)


def main() -> None:
    """Run the aggregate synthetic-data audit from the command line.

    Returns
    -------
    None
        Aggregate JSON summaries are printed to standard output.
    """

    parser = argparse.ArgumentParser(
        description="Audit public synthetic data using aggregate summaries.",
    )
    parser.add_argument(
        "--compare-private",
        action="store_true",
        help="Also use ignored local raw data for aggregate and overlap checks.",
    )
    arguments = parser.parse_args()

    synthetic_booking, synthetic_attendance = load_data(use_synthetic=True)
    report: dict[str, object] = {
        "synthetic": build_calibration_summary(
            synthetic_booking,
            synthetic_attendance,
        )
    }

    if arguments.compare_private:
        original_booking, original_attendance = load_data(use_synthetic=False)
        report["original"] = build_calibration_summary(
            original_booking,
            original_attendance,
        )
        report["privacy_sanity_checks"] = run_privacy_sanity_checks(
            original_booking,
            original_attendance,
            synthetic_booking,
            synthetic_attendance,
        )

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
