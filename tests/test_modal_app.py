from pathlib import Path

import modal_app
from src.main import app as existing_application


def test_modal_asgi_wrapper_validates_package_and_returns_existing_app(monkeypatch):
    validated_paths = []

    monkeypatch.setattr(
        "src.mlops.runtime.validate_production_startup",
        lambda path: validated_paths.append(path),
    )

    raw_wrapper = modal_app.fastapi_app.get_raw_f()
    assert raw_wrapper() is existing_application
    assert validated_paths == ["/app/build/model"]


def test_modal_image_definition_is_inference_only():
    source = Path(modal_app.__file__).read_text(encoding="utf-8")
    requirements = Path("requirements-serving.txt").read_text(encoding="utf-8")

    assert 'python_version="3.12"' in source
    assert 'remote_path="/app/modal_app.py"' in source
    assert 'remote_path="/app/build/model"' in source
    assert 'include_source=False' in source
    assert 'modal.Secret.from_name(DATABASE_SECRET_NAME)' in source
    assert 'DATABASE_SECRET_NAME = "medical-insurance-database"' in source
    assert (
        'PRODUCTION_FRONTEND_ORIGIN = "https://medical-insurance-cost.vercel.app"'
        in source
    )
    assert 'env={"CORS_ALLOWED_ORIGINS": CORS_ALLOWED_ORIGINS}' in source
    assert "secrets=[database_secret]" in source
    assert "MLFLOW_TRACKING_PASSWORD" not in source
    assert '"training"' not in source
    assert '"deployment.py"' not in source
    assert '"registry.py"' not in source
    assert 'PROJECT_ROOT / "templates"' not in source
    assert "jinja2" not in requirements.lower()
    assert "python-multipart" not in requirements.lower()
    assert "dagshub" not in requirements.lower()
    assert "modal" not in requirements.lower()
