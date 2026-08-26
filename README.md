# Medical Insurance Cost Prediction

![CI](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml/badge.svg?branch=main&event=push)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/github/license/TumeloKonaite/Medical-Insurance-Cost)

Train regression models locally and serve medical-insurance charge predictions through an HTML form or a JSON API.

![Demo preview](docs/demo.png)

## Architecture

The application uses explicit, one-way dependencies:

```text
FastAPI routes
    -> Pydantic request/response schemas
    -> PredictionService
    -> ArtifactRepository protocol
    -> LocalArtifactRepository
```

- `src.api` contains application composition and lean HTTP routes.
- `src.schemas` owns the shared form and JSON validation contract.
- `src.services` owns inference orchestration.
- `src.repositories` is the only layer that reads or writes serialized model artifacts.
- `src.training` contains data ingestion, transformation, and model selection.

The application creates one prediction service per process. A single fitted scikit-learn pipeline containing preprocessing and regression is loaded lazily on the first prediction and cached for that service lifecycle.

## Local setup

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

Install the application and development dependencies:

```bash
uv sync --extra dev
```

Alternatively, install the runtime requirements with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train and create local artifacts

Predictions require `artifacts/model.pkl`. This one artifact is the complete fitted pipeline and accepts the six raw request fields directly. Create it from the included dataset before using either prediction endpoint:

```bash
uv run python scripts/run_pipeline.py
```

The training pipeline creates its data splits and serialized artifacts under `artifacts/`, which is intentionally ignored by Git.

MLflow tracking is disabled by default, so the command above is entirely local and does not import or contact MLflow during training. To record the experiment in the local `mlruns/` file store instead:

```bash
ENABLE_MLFLOW_TRACKING=true \
MLFLOW_EXPERIMENT_NAME=medical-insurance-cost \
uv run python scripts/run_pipeline.py

MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui
```

Open <http://localhost:5000> to inspect candidate parameters and metrics, the selected-model metrics, raw-feature signature, input example, and logged pipeline. The file-store opt-in keeps this flow compatible with MLflow 3.15 and later. Set `MLFLOW_TRACKING_URI` to choose another backend. Model registration is intentionally disabled; `ENABLE_MODEL_REGISTRATION=false` is the supported setting for this version.

Tracking settings:

- `ENABLE_MLFLOW_TRACKING` defaults to `false`.
- `MLFLOW_TRACKING_URI` is optional; enabled tracking defaults to the local `./mlruns` store.
- `MLFLOW_EXPERIMENT_NAME` defaults to `medical-insurance-cost`.
- `ENABLE_MODEL_REGISTRATION` must remain `false` for this version.
- `MLFLOW_REGISTERED_MODEL_NAME` defaults to `medical-insurance-cost` and is reserved for later registration work.

## Run the API

```bash
uv run uvicorn src.main:app --reload
```

Open the HTML form at <http://localhost:8000/> or the interactive API documentation at <http://localhost:8000/docs>.

Health check:

```bash
curl http://localhost:8000/health
```

HTML form prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "age=29" \
  -d "sex=female" \
  -d "bmi=27.4" \
  -d "children=2" \
  -d "smoker=no" \
  -d "region=southeast"
```

JSON prediction:

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"age":29,"sex":"female","bmi":27.4,"children":2,"smoker":"no","region":"southeast"}'
```

Successful JSON responses use this schema:

```json
{
  "predicted_charges": 12345.67,
  "currency": "USD"
}
```

Valid categorical values are:

- `sex`: `female`, `male`
- `smoker`: `yes`, `no`
- `region`: `northeast`, `northwest`, `southeast`, `southwest`

## Docker

Train the artifacts locally first, then build and run the image:

```bash
docker build -t insurance-cost-api .
docker run --rm -p 8000:8000 insurance-cost-api
```

Or use Compose:

```bash
docker compose up --build
```

## Development

```bash
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Equivalent Make targets are available:

```bash
make pipeline
make run
make test
make lint
```

## Project structure

```text
src/
├── api/
│   ├── dependencies.py
│   └── routes/
├── repositories/
├── schemas/
├── services/
├── training/
├── exceptions.py
└── main.py
```

- Dataset: `Data/medical_insurance.csv`
- Model card: `docs/MODEL_CARD.md`
- License: `LICENSE`
