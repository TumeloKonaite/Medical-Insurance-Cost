"""Experiment tracking and model-registry integration."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.mlops.config import MLflowConfig, MlflowConfig

__all__ = ["MLflowConfig", "MlflowConfig"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.mlops.config import MLflowConfig, MlflowConfig

        return {"MLflowConfig": MLflowConfig, "MlflowConfig": MlflowConfig}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
