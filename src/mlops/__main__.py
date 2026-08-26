from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from src.exceptions import ApplicationError
from src.mlops.config import MlflowConfig
from src.mlops.deployment import prepare_deployment
from src.mlops.registry import (
    inspect_version,
    promote_version,
    resolve_alias,
    verify_alias_and_numeric,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MLflow model-registry operations")
    commands = parser.add_subparsers(dest="command", required=True)

    resolve = commands.add_parser(
        "resolve-model", help="Resolve a validated alias to an immutable model version"
    )
    _model_name_argument(resolve)
    resolve.add_argument("--alias", required=True)
    _output_argument(resolve)
    resolve.set_defaults(handler=_resolve)

    inspect = commands.add_parser(
        "inspect-model", help="Inspect one exact registered model version"
    )
    _model_name_argument(inspect)
    inspect.add_argument("--version", required=True)
    _output_argument(inspect)
    inspect.set_defaults(handler=_inspect)

    promote = commands.add_parser(
        "promote-model", help="Validate a numeric version and explicitly move an alias"
    )
    _model_name_argument(promote)
    promote.add_argument("--version", required=True)
    promote.add_argument("--alias", required=True)
    promote.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement that this command changes promotion state",
    )
    _output_argument(promote)
    promote.set_defaults(handler=_promote)

    verify = commands.add_parser(
        "verify-model", help="Compare alias and numeric model loading and predictions"
    )
    _model_name_argument(verify)
    verify.add_argument("--alias", required=True)
    _output_argument(verify)
    verify.set_defaults(handler=_verify)

    prepare = commands.add_parser(
        "prepare-deployment",
        help="Validate and package one exact numeric registered model version",
    )
    prepare.add_argument("--model-uri", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--expected-run-id", required=True)
    prepare.add_argument("--expected-pipeline-sha256", required=True)
    _output_argument(prepare)
    prepare.set_defaults(handler=_prepare_deployment)

    validate = commands.add_parser(
        "validate-deployment",
        help="Validate a completed local deployment package without registry access",
    )
    validate.add_argument("--package-dir", required=True)
    _output_argument(validate)
    validate.set_defaults(handler=_validate_deployment)
    return parser


def _model_name_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-name", required=True)


def _output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", choices=("json", "text"), default="text")


def _resolve(args: argparse.Namespace, config: MlflowConfig) -> dict[str, Any]:
    return resolve_alias(config, args.model_name, args.alias).as_dict()


def _inspect(args: argparse.Namespace, config: MlflowConfig) -> dict[str, Any]:
    return inspect_version(config, args.model_name, args.version).as_dict()


def _promote(args: argparse.Namespace, config: MlflowConfig) -> dict[str, Any]:
    return promote_version(
        config,
        args.model_name,
        args.version,
        args.alias,
        confirmed=args.confirm,
    ).as_dict()


def _verify(args: argparse.Namespace, config: MlflowConfig) -> dict[str, Any]:
    return verify_alias_and_numeric(config, args.model_name, args.alias)


def _prepare_deployment(
    args: argparse.Namespace, config: MlflowConfig
) -> dict[str, Any]:
    return prepare_deployment(
        config=config,
        model_uri=args.model_uri,
        output_dir=args.output_dir,
        expected_run_id=args.expected_run_id,
        expected_pipeline_sha256=args.expected_pipeline_sha256,
    ).as_dict()


def _validate_deployment(
    args: argparse.Namespace, _config: MlflowConfig
) -> dict[str, Any]:
    from src.mlops.runtime import validate_local_package

    return dict(validate_local_package(args.package_dir).metadata)


def _render(payload: dict[str, Any], output: str) -> None:
    if output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    handler: Callable[[argparse.Namespace, MlflowConfig], dict[str, Any]] = args.handler
    try:
        payload = handler(args, MlflowConfig.from_env())
    except ApplicationError as exc:
        parser.exit(2, f"error: {exc}\n")
    _render(payload, args.output)


if __name__ == "__main__":
    main()
