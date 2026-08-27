# Local development guide

[Back to the project README](../README.md)

This guide covers local training, API and frontend development, database setup,
Docker, and the test suite.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 24+
- npm 10+

## Install dependencies

From the repository root, install the backend and development dependencies:

```bash
uv sync --locked --extra dev --extra monitoring
```

Install the frontend from its committed lockfile:

```bash
cd frontend
npm ci
cd ..
```

Do not commit `.env` or `frontend/.env` files. Values prefixed with `VITE_` are
included in browser JavaScript and must never contain secrets.

## Train a local model

The dataset is stored at `Data/medical_insurance.csv`. Run the deterministic
training pipeline to create the train/test splits and fitted scikit-learn
pipeline under the ignored `artifacts/` directory:

```bash
uv run python scripts/run_pipeline.py
```

MLflow tracking is disabled by default, so this command runs entirely locally.
To inspect the experiment in a local MLflow store:

```bash
ENABLE_MLFLOW_TRACKING=true \
MLFLOW_EXPERIMENT_NAME=medical-insurance-cost \
uv run python scripts/run_pipeline.py

MLFLOW_ALLOW_FILE_STORE=true uv run mlflow ui
```

Open <http://localhost:5000> to inspect candidate parameters, metrics, the
selected pipeline, raw-feature signature, and input example.

## Optional prediction-event database

Predictions remain available when `DATABASE_URL` is unset. Configure it to store
successful predictions and their monitoring outbox records in Neon PostgreSQL.
Use the pooled connection string, whose hostname contains `-pooler`, and keep it
in an ignored `.env` file:

```dotenv
DATABASE_URL=postgresql://<role>:<password>@<endpoint>-pooler.<region>.aws.neon.tech/<database>?sslmode=require
```

Apply the checked-in Alembic migrations:

```bash
uv run --env-file .env alembic upgrade head
```

The application never creates or changes tables during startup. Review the
selected database and its backups before intentionally downgrading migrations.

## Run the API

```bash
uv run --env-file .env uvicorn src.main:app --reload --port 8000
```

If no database is required locally, omit `--env-file .env`. Open the interactive
API documentation at <http://localhost:8000/docs>.

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict-json \
  -H "Content-Type: application/json" \
  -d '{"age":29,"sex":"female","bmi":27.4,"children":2,"smoker":"no","region":"southeast"}'
```

Successful responses follow this schema:

```json
{
  "predicted_charges": 12345.67,
  "currency": "USD"
}
```

Accepted categorical values are:

- `sex`: `female`, `male`
- `smoker`: `yes`, `no`
- `region`: `northeast`, `northwest`, `southeast`, `southwest`

## Run the frontend

Start the API first, then use a second terminal:

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. In development, Vite proxies `/predict-json` to
`http://localhost:8000`, so the frontend does not need a hard-coded local API
URL.

### Vercel production settings

The production Vercel project uses:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Node.js Version | `24.x` |

Configure this public build-time value for Production and Preview:

```dotenv
VITE_API_BASE_URL=https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run
```

Redeploy after changing it. The frontend validates the URL, removes trailing
slashes, and appends `/predict-json`. It rejects credentials, query strings,
fragments, and insecure non-local HTTP URLs.

### Backend CORS allowlist

FastAPI accepts a comma-separated list of exact browser origins:

```dotenv
CORS_ALLOWED_ORIGINS=http://localhost:5173,https://medical-insurance-cost.vercel.app
```

Wildcard origins and origins with credentials, paths, queries, or fragments are
rejected. Generated Vercel Preview domains are not automatically allowed. Add a
specific trusted preview origin and redeploy Modal when live preview predictions
are required.

## Docker

Create `artifacts/model.pkl` with the local pipeline before building the
development image:

```bash
docker build -t insurance-cost-api .
docker run --rm -p 8000:8000 insurance-cost-api
```

Or use Compose:

```bash
docker compose up --build
```

## Run the checks

Backend:

```bash
uv run --extra dev --extra monitoring ruff check .
uv run --extra dev --extra monitoring pytest
```

Frontend:

```bash
cd frontend
npm run lint
npm test
npm run build
```

Equivalent Make targets include `make pipeline`, `make run`, `make test`, and
`make lint`.
