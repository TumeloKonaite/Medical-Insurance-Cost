from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parent
APP_NAME = "medical-insurance-cost"
PACKAGE_DIR = PROJECT_ROOT / "build" / "model"
DATABASE_SECRET_NAME = "medical-insurance-database"

app = modal.App(APP_NAME)
# DATABASE_URL is injected at runtime and is never baked into the image.
database_secret = modal.Secret.from_name(DATABASE_SECRET_NAME)

# Copy only inference code, templates, and the verified package. Training, registry,
# datasets, tests, credentials, and local caches never enter the production image.
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
        str(PROJECT_ROOT / "templates"),
        remote_path="/app/templates",
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


def _load_fastapi_application():
    from src.mlops.runtime import validate_production_startup

    validate_production_startup("/app/build/model")

    from src.main import app as application

    return application


@app.function(
    image=image,
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
