from __future__ import annotations

import argparse
import json
import logging

from src.monitoring.baseline import BaselineUploadError, upload_baseline
from src.monitoring.config import (
    ArizeExportConfig,
    MonitoringConfigurationError,
)
from src.monitoring.exporter import (
    ExportInfrastructureError,
    ExportRunFailed,
    run_exporter_from_env,
)
from src.monitoring.ground_truth import GroundTruthError, record_actual_from_env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.monitoring")
    commands = parser.add_subparsers(dest="command", required=True)

    actual = commands.add_parser(
        "record-actual", help="Record delayed ground truth by request ID."
    )
    actual.add_argument("--request-id", required=True)
    actual.add_argument("--actual-charges", required=True)

    baseline = commands.add_parser(
        "upload-baseline", help="Upload a validation baseline to Arize."
    )
    baseline.add_argument("--test-data", required=True)
    baseline.add_argument("--model-package", required=True)

    commands.add_parser("export", help="Run one outbox export invocation.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parser().parse_args()
    try:
        if args.command == "record-actual":
            result = record_actual_from_env(
                request_id_text=args.request_id,
                actual_charges_text=args.actual_charges,
            )
        elif args.command == "upload-baseline":
            result = upload_baseline(
                config=ArizeExportConfig.from_environment(),
                test_data=args.test_data,
                model_package=args.model_package,
            )
        else:
            result = run_exporter_from_env()
    except ExportRunFailed as exc:
        print(json.dumps(exc.summary.as_dict(), sort_keys=True))
        return 1
    except (
        MonitoringConfigurationError,
        ExportInfrastructureError,
        GroundTruthError,
        BaselineUploadError,
    ) as exc:
        print(f"Monitoring command failed: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
