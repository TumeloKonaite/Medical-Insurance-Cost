class ApplicationError(Exception):
    """Base class for expected application failures."""


class ArtifactRepositoryError(ApplicationError):
    """Base class for artifact persistence failures."""


class ArtifactNotFoundError(ArtifactRepositoryError):
    """Raised when a required artifact does not exist."""


class ArtifactLoadError(ArtifactRepositoryError):
    """Raised when an artifact cannot be safely loaded or validated."""


class ArtifactSaveError(ArtifactRepositoryError):
    """Raised when an artifact cannot be persisted."""


class PredictionServiceError(ApplicationError):
    """Base class for prediction failures."""


class ArtifactUnavailableError(PredictionServiceError):
    """Raised when prediction artifacts are unavailable."""


class PredictionError(PredictionServiceError):
    """Raised when preprocessing or inference fails."""


class TrainingError(ApplicationError):
    """Raised when the training pipeline fails."""


class MlflowConfigurationError(TrainingError):
    """Raised when MLflow settings are incomplete or unsafe."""


class ModelRegistryError(ApplicationError):
    """Raised when a model-registry operation cannot be completed safely."""


class DeploymentError(ApplicationError):
    """Raised when an immutable deployment package cannot be built or validated."""
