"""Public interface for reusable modeling-stage utilities.

The modeling package supports the final notebook stages of the project.
It contains small, focused operations that need direct unit-test coverage.
Model comparison and scientific interpretation remain visible in notebooks.
No model training or evaluation is performed by this package yet.
"""

from .splitting import (
    PREDICTION_IDENTIFIER_COLUMNS,
    assign_temporal_split,
    validate_prediction_identity,
    validate_prediction_time,
)

__all__ = [
    "PREDICTION_IDENTIFIER_COLUMNS",
    "assign_temporal_split",
    "validate_prediction_identity",
    "validate_prediction_time",
]
