"""Canonical model input and target contract."""

FEATURE_COLUMNS = (
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
)

TARGET_COLUMN = "charges"

# Bump these values whenever a deployment-facing input or output contract changes.
FEATURE_SCHEMA_VERSION = "1"
PREDICTION_CONTRACT_VERSION = "1"
