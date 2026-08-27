from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "medical-insurance-cost"
PACKAGE_DIR = PROJECT_ROOT / "build" / "model"
DATABASE_SECRET_NAME = "medical-insurance-database"
ARIZE_SECRET_NAME = "medical-insurance-arize"
PRODUCTION_FRONTEND_ORIGIN = "https://medical-insurance-cost.vercel.app"
CORS_ALLOWED_ORIGINS = (
    f"http://localhost:5173,{PRODUCTION_FRONTEND_ORIGIN}"
)

app = modal.App(APP_NAME)
# DATABASE_URL is injected at runtime and is never baked into the image.
database_secret = modal.Secret.from_name(DATABASE_SECRET_NAME)
arize_secret = modal.Secret.from_name(ARIZE_SECRET_NAME)

# Copy only inference code and the verified package. Training, registry, datasets,
# tests, credentials, frontend assets, and local caches never enter the image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        str(PROJECT_ROOT / "requirements-serving.txt")
    )
    .add_local_file(
        str(PROJECT_ROOT / "modal_app.py"),
        remote_path="/app/modal_app.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "__init__.py"),
        remote_path="/app/src/__init__.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "main.py"),
        remote_path="/app/src/main.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "exceptions.py"),
        remote_path="/app/src/exceptions.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "model_contract.py"),
        remote_path="/app/src/model_contract.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "paths.py"),
        remote_path="/app/src/paths.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "database.py"),
        remote_path="/app/src/database.py",
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "api"),
        remote_path="/app/src/api",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "schemas"),
        remote_path="/app/src/schemas",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "services"),
        remote_path="/app/src/services",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "repositories"),
        remote_path="/app/src/repositories",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "models"),
        remote_path="/app/src/models",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "mlops" / "__init__.py"),
        remote_path="/app/src/mlops/__init__.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "mlops" / "runtime.py"),
        remote_path="/app/src/mlops/runtime.py",
        copy=True,
    )
    .add_local_dir(
        str(PACKAGE_DIR),
        remote_path="/app/build/model",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .env({"MODEL_PACKAGE_DIR": "/app/build/model"})
    .workdir("/app")
)

# This image contains the outbox exporter, but no model package or API server.
arize_export_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(
        str(PROJECT_ROOT / "requirements-monitoring.txt")
    )
    .add_local_file(
        str(PROJECT_ROOT / "modal_app.py"),
        remote_path="/app/modal_app.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "__init__.py"),
        remote_path="/app/src/__init__.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "database.py"),
        remote_path="/app/src/database.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "model_contract.py"),
        remote_path="/app/src/model_contract.py",
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "models"),
        remote_path="/app/src/models",
        copy=True,
        ignore=["**/__pycache__", "**/*.pyc"],
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "monitoring" / "__init__.py"),
        remote_path="/app/src/monitoring/__init__.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "monitoring" / "config.py"),
        remote_path="/app/src/monitoring/config.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "monitoring" / "arize_client.py"),
        remote_path="/app/src/monitoring/arize_client.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "monitoring" / "outbox.py"),
        remote_path="/app/src/monitoring/outbox.py",
        copy=True,
    )
    .add_local_file(
        str(PROJECT_ROOT / "src" / "monitoring" / "exporter.py"),
        remote_path="/app/src/monitoring/exporter.py",
        copy=True,
    )
    .workdir("/app")
)


def _load_fastapi_application():
    from src.mlops.runtime import validate_production_startup

    validate_production_startup("/app/build/model")

    from src.main import app as application

    return application


@app.function(
    image=image,
    env={"CORS_ALLOWED_ORIGINS": CORS_ALLOWED_ORIGINS},
    secrets=[database_secret],
    cpu=1.0,
    timeout=600,
    min_containers=0,
    scaledown_window=300,
    include_source=False,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    return _load_fastapi_application()


@app.function(
    image=arize_export_image,
    secrets=[database_secret, arize_secret],
    schedule=modal.Cron("5 * * * *"),
    timeout=600,
    include_source=False,
)
def export_predictions_to_arize():
    import logging

    from src.monitoring.exporter import ExportRunFailed, run_exporter_from_env

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("src.monitoring").setLevel(logging.INFO)
    try:
        summary = run_exporter_from_env()
    except ExportRunFailed as exc:
        logging.getLogger(__name__).error(
            "arize_export_invocation_failed summary=%s", exc.summary.as_dict()
        )
        raise RuntimeError("The Arize export invocation failed.") from None
    logging.getLogger(__name__).info(
        "arize_export_invocation_complete summary=%s", summary
    )
    return summary
