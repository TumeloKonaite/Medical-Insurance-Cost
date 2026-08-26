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
    assert "modal.Secret" not in source
    assert '"training"' not in source
    assert '"deployment.py"' not in source
    assert '"registry.py"' not in source
    assert "dagshub" not in requirements.lower()
    assert "modal" not in requirements.lower()
