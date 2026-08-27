# Medical Insurance Cost Prediction

![CI](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml/badge.svg?branch=main&event=push)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/github/license/TumeloKonaite/Medical-Insurance-Cost)

Train regression models locally and serve medical-insurance charge predictions through a React interface and a FastAPI JSON API.

Live demo:
<https://medical-insurance-cost.vercel.app>

Production backend URL:
<https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run>

Production API docs:
<https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run/docs>

![Demo preview](docs/demo.png)

## Architecture

The frontend and backend are independent applications with explicit, one-way dependencies:

```text
React + Vite frontend
    -> POST /predict-json
    -> FastAPI routes
    -> Pydantic request/response schemas
    -> PredictionService
    -> ArtifactRepository protocol
    -> LocalArtifactRepository (development) or PackagedMlflowRepository (production)

Successful prediction
    -> PredictionEventService (fail-open boundary)
    -> PredictionEventRepository
    -> PostgreSQL
```

- `frontend/` contains the React + Vite TypeScript application, browser validation,
  and typed API client.
- `src.api` contains application composition and lean HTTP routes.
- `src.schemas` owns the authoritative JSON validation contract.
- `src.services` owns inference orchestration.
- `src.repositories` is the only layer that reads or writes serialized model artifacts.
- `PredictionEventService` builds monitoring events independently of inference; its
  repository is the only layer that writes prediction data.
- `src.training` contains data ingestion, transformation, and model selection.

The application creates one prediction service per process. A single fitted scikit-learn pipeline containing preprocessing and regression is loaded lazily on the first prediction and cached for that service lifecycle. Modal startup validates and loads the baked MLflow package before returning the ASGI application, and the production repository reuses that process-cached model.

## Local setup

Requirements:

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Node.js 24+
- npm 10+

Install the application and development dependencies:

```bash
uv sync --extra dev
```

Alternatively, install the runtime requirements with pip:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install the frontend from its committed lockfile:

```bash
cd frontend
npm ci
```

Do not commit a frontend `.env` file. Values prefixed with `VITE_` are bundled
into browser code and must never contain secrets.

## Prediction event storage

Successful `/predict-json` predictions are stored as typed PostgreSQL rows.
These events provide the basis for measuring prediction
volume, input and output distributions, model-version usage, inference latency,
and future drift or accuracy once actual charges are available. Invalid requests
and failed inference attempts are not stored.

Persistence is deliberately fail-open. A database outage or rejected insert is
logged using only the generated request ID and exception type, and the successful
prediction is still returned. Database errors and credentials are never included
in an API response. `request_id` is unique, and inserts ignore conflicts so a
retry with the same ID cannot create another row. The application does not create
or alter tables during startup.

The `prediction_events` schema is:

| Column | Type | Notes |
| --- | --- | --- |
| `id` | UUID | Primary key |
| `request_id` | UUID | Unique and indexed |
| `created_at` | Timestamp with time zone | Indexed |
| `source` | String | `json` for the current API |
| `age` | Integer | Model feature |
| `sex` | String | Model feature |
| `bmi` | Numeric | Model feature |
| `children` | Integer | Model feature |
| `smoker` | String | Model feature |
| `region` | String | Model feature |
| `predicted_charges` | Numeric | Successful model output |
| `model_version` | String | Indexed; `local` locally or the packaged MLflow version |
| `prediction_contract_version` | String | Deployment contract version |
| `inference_latency_ms` | Numeric | Model call duration only |
| `actual_charges` | Nullable numeric | Reserved for a later accuracy workflow |
| `actual_recorded_at` | Nullable timestamp with time zone | When actual charges were supplied |

### Configure Neon and apply migrations

Create a Neon project and database, then copy the pooled connection string from
the Neon dashboard. Pooled hosts contain `-pooler` in the hostname. Put it in a
local `.env` file or the deployment platform's secret store—never in source
control:

```dotenv
DATABASE_URL=postgresql://<role>:<password>@<endpoint>-pooler.<region>.aws.neon.tech/<database>?sslmode=require
```

The application requires an encrypted psycopg SSL mode, upgrades missing or
unsafe modes to `sslmode=require`, preserves stricter verification modes, and
rejects a non-pooled Neon hostname. Load the environment and apply the checked-in
Alembic migrations from the repository root:

```bash
set -a
source .env
set +a
uv run alembic upgrade head
```

That command is the same for a development Neon branch and the deployed Neon
database; select the target solely through `DATABASE_URL`. Review the target and
backups before running a downgrade. To revert the initial schema intentionally:

```bash
uv run alembic downgrade base
```

After migration, start the API with the same `DATABASE_URL` environment variable.
If it is unset, inference remains available but event persistence is disabled.

### Privacy and intended use

Store only the six model features and monitoring metadata listed above. Do not
add names, email addresses, account identifiers, free-form request bodies, or
other personally identifying information. Restrict database access, rotate Neon
credentials, use separate roles and branches by environment, and define retention
rules before collecting any data beyond this demonstration.

This repository and its prediction-event data use synthetic/demo data only. The
application is not intended for real medical use, medical decisions, diagnosis,
insurance underwriting, or production availability, and it has no HIPAA or other
regulatory certification.

## Arize monitoring

Arize AX receives asynchronous copies for observability; Neon remains the durable
source of truth. `/predict-json` writes a prediction and a pending
`arize_export_events` row in one Neon transaction, then returns without importing
or contacting Arize. An hourly Modal function claims outbox rows, groups them by
exact numeric MLflow model version and event type, and batch-uploads only the six
approved features and monitoring fields. Delivery is at least once and always
reuses `request_id` as the Arize prediction ID.

### Configure Arize AX and Modal

1. Create an [Arize AX account](https://arize.com/) and a space. This project uses
   model name `medical-insurance-cost`, numeric model type, production data for
   live predictions, and validation data for the held-out baseline.
2. In the Arize space, create or copy an API key and copy the space ID. Put only
   placeholder values in `.env.example`; load real values from an ignored `.env`
   or secret manager:

   ```dotenv
   ARIZE_API_KEY=<arize-api-key>
   ARIZE_SPACE_ID=<arize-space-id>
   ARIZE_MODEL_NAME=medical-insurance-cost
   ARIZE_EXPORT_BATCH_SIZE=500
   ```

3. Create the dedicated Modal secret. It is attached only to the exporter, never
   the FastAPI serving function:

   ```bash
   uv run --all-extras --env-file .env bash -c '
     test -n "$ARIZE_API_KEY" && test -n "$ARIZE_SPACE_ID" || {
       echo "Arize settings are missing"; exit 1;
     }
     modal secret create --force medical-insurance-arize \
       ARIZE_API_KEY="$ARIZE_API_KEY" \
       ARIZE_SPACE_ID="$ARIZE_SPACE_ID"
   '
   ```

4. Apply the outbox migration to the intended Neon database before deployment:

   ```bash
   uv run alembic upgrade head
   ```

   The migration also queues existing prediction events once. New prediction
   events and their outbox rows commit atomically. Persistence remains fail-open
   for inference when Neon is disabled or unavailable.

### Upload the validation baseline

Install the locked monitoring dependency and build the immutable production model
package first. Baseline upload is an explicit release step and never runs during
API startup:

```bash
uv sync --locked --extra monitoring
uv run --extra monitoring --env-file .env python -m src.monitoring upload-baseline \
  --test-data artifacts/test.csv \
  --model-package build/model
```

The command validates the packaged model, its six-field MLflow signature, and
`deployment_metadata.json`; validates the exact CSV columns
`age,sex,bmi,children,smoker,region,charges`; generates predictions with the
packaged pipeline; and uploads stable row IDs, predictions, actuals, the exact
numeric model version, and MLflow run ID as the batch ID to Arize validation.
It refuses malformed datasets, packages, signatures, metadata, targets, or
non-finite predictions.

### Deploy and run the exporter

Deploying the existing Modal app activates the UTC cron schedule at minute 5 of
every hour:

```bash
uv sync --locked --extra deployment
uv run modal deploy modal_app.py
```

Run the same function manually after setup or during an operational check:

```bash
uv run modal run modal_app.py::export_predictions_to_arize
```

For a local one-shot run against the configured Neon and Arize environments:

```bash
uv run --extra monitoring --env-file .env python -m src.monitoring export
```

The exporter fails closed when `DATABASE_URL`, `ARIZE_API_KEY`, or
`ARIZE_SPACE_ID` is missing. This does not affect FastAPI startup. Each invocation
logs a sanitized summary with records claimed, sent, retried, failed, remaining
backlog, and oldest pending age. Failed batches are returned to `pending` with
bounded exponential backoff; rows abandoned in `processing` are recovered after
`ARIZE_CLAIM_STALE_MINUTES` (30 by default). A delivery failure is re-raised only
after retry state is committed, so Modal marks the invocation failed.

### Record delayed ground truth

There is no public actual-label endpoint. An operator with Neon access can record
an actual charge by the prediction's stored request ID:

```bash
uv run python -m src.monitoring record-actual \
  --request-id <uuid> \
  --actual-charges 12345.67
```

The command requires `DATABASE_URL`, accepts only finite non-negative values,
updates the prediction and creates an `actual` outbox row in one transaction. The
same value is idempotent; a different existing value is rejected. Arize joins the
delayed actual to the original prediction through the unchanged `request_id`.

### Operate monitors, backlog, and credentials

Open the [medical-insurance-cost model directly in Arize AX][arize-model]. From
that model page, use the top-level **Monitor** and **Dashboard** tabs. This
space-specific link avoids being redirected to the generic Arize home or LLM
tracing pages.

[arize-model]: https://app.arize.com/organizations/QWNjb3VudE9yZ2FuaXphdGlvbjo0OTE3OTo2RjBo/spaces/U3BhY2U6NTI3MzA6SWZIcw==/models/modelName/medical-insurance-cost?selectedTab=performance

Inspect recent scheduled-function failures and sanitized summaries in the Modal
dashboard or CLI:

```bash
uv run modal app logs medical-insurance-cost --since 24h --timestamps
```

The summary is the primary backlog signal. For a database-side aggregate that
does not expose features or charges, use:

```sql
SELECT status, event_type, count(*) AS records, min(created_at) AS oldest
FROM arize_export_events
WHERE status <> 'sent'
GROUP BY status, event_type;
```

Configure these initial monitors in the Arize UI:

- Prediction count and no-data alerts.
- Prediction-distribution drift.
- Drift on the most influential approved features.
- Numeric feature range and data-quality checks.
- Unexpected categorical values.
- p95 inference latency.
- Traffic grouped by model version.
- MAE, RMSE, and R² after actual labels arrive.
- Actual-label coverage and delay, if supported through a custom metric.

This is a low-volume demo, so prefer daily or multi-day evaluation windows over
noisy hourly statistical alerts. Arize Free retention and usage limits can
change; check the current [Arize pricing](https://arize.com/pricing/) and retain
the complete history in Neon.

To rotate the API key, create a replacement in Arize, overwrite the Modal secret,
redeploy, manually run one export, and then revoke the old key:

```bash
uv run modal secret create --force medical-insurance-arize \
  ARIZE_API_KEY="$NEW_ARIZE_API_KEY" \
  ARIZE_SPACE_ID="$ARIZE_SPACE_ID"
uv run modal deploy modal_app.py
uv run modal run modal_app.py::export_predictions_to_arize
```

Never log or commit Arize credentials. Do not send names, email addresses, IP
addresses, database row IDs, raw bodies, free-form text, SHAP values, traces, or
training candidates. The six model features and charge labels can still be
sensitive; this integration is approved only for the repository's synthetic/demo
data. CI uses fake clients and does not contact Arize or require credentials.

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

Open <http://localhost:5000> to inspect candidate parameters and metrics, the selected-model metrics, raw-feature signature, input example, and logged pipeline. The file-store opt-in keeps this flow compatible with MLflow 3.15 and later. Set `MLFLOW_TRACKING_URI` to choose another backend; registration remains off unless explicitly enabled.

## DagsHub experiment tracking and model registry

The training code uses the same MLflow implementation for a local store and DagsHub; only environment and authentication settings differ. DagsHub runs are tagged with `tracking_backend=dagshub`. Registration occurs only after candidate evaluation, finite-metric validation, a raw-feature smoke prediction, and checks for the logged signature and input example. Only the selected pipeline is registered.

### Configure DagsHub

Every DagsHub repository has an MLflow endpoint at `https://dagshub.com/<owner>/<repository>.mlflow`. Find the exact value from the repository's **Remote** menu under **MLflow**, or append `.mlflow` to the repository URL. See the [DagsHub MLflow tracking guide](https://dagshub.com/docs/integration_guide/mlflow_tracking/) for the current UI.

Create an access token from [DagsHub user token settings](https://dagshub.com/user/settings/tokens). The account must have contributor access to the repository. Copy `.env.example` to `.env`, replace every placeholder locally, and load it into the shell before running a command:

```bash
cp .env.example .env
set -a
source .env
set +a
```

The expected settings are:

```dotenv
ENABLE_MLFLOW_TRACKING=true
ENABLE_MODEL_REGISTRATION=true
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repository>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-access-token>
MLFLOW_EXPERIMENT_NAME=medical-insurance-cost
MLFLOW_REGISTERED_MODEL_NAME=medical-insurance-cost
```

`.env` is ignored by Git. Treat `MLFLOW_TRACKING_PASSWORD` as a secret: do not place it in the URI, shell history, logs, screenshots, commits, or issue text. The application uses MLflow's standard environment-variable authentication and deliberately fails instead of choosing the local store when credentials are present without a remote URI. Remote authentication and configuration errors are sanitized.

### Train and review a candidate

With the environment loaded, run:

```bash
uv run python scripts/run_pipeline.py
```

When both feature flags are true, the command logs one experiment and registers only the selected model as the next numeric version of `medical-insurance-cost`. It prints the run ID, source model URI, Git commit, dataset checksum, numeric version, and registered pipeline checksum. It does not assign `champion`.

Open the configured `.mlflow` URL in a browser. In the experiment run, compare candidate and selected metrics and inspect the parameters, Git and dataset lineage, signature, and input example. In **Models**, open `medical-insurance-cost` and its new version. New versions have `validation_status=candidate` plus these deployment-lineage tags:

- `training_run_id`
- `source_commit_sha`
- `dataset_sha256`
- `feature_schema_version`
- `prediction_contract_version`
- `selected_model`
- `selection_metric`
- `validation_status`
- `pipeline_sha256`

Inspect the same metadata from the terminal with an exact numeric version:

```bash
uv run python -m src.mlops inspect-model \
  --model-name medical-insurance-cost \
  --version 7 \
  --output json
```

### Promote, resolve, and verify

After the review succeeds, promotion is manual. In the DagsHub MLflow UI, set the version tag `validation_status=validated`, then assign or move the `champion` alias to that exact version. MLflow aliases are mutable by design; the [MLflow registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow) describes their UI and API behavior.

The explicit CLI performs those two writes and requires the exact version plus confirmation:

```bash
uv run python -m src.mlops promote-model \
  --model-name medical-insurance-cost \
  --version 7 \
  --alias champion \
  --confirm \
  --output json
```

Resolve the human-facing alias before deployment:

```bash
uv run python -m src.mlops resolve-model \
  --model-name medical-insurance-cost \
  --alias champion \
  --output json
```

The resolver fails for a missing alias, nonnumeric version, incomplete lineage, malformed checksums, or any status other than `validated`. Its `model_uri` is always immutable, for example `models:/medical-insurance-cost/7`. Give that exact URI to deployment; never deploy `models:/medical-insurance-cost@champion` or `models:/medical-insurance-cost/latest`.

Before a deployment, verify that alias and numeric loading identify the same run and produce compatible finite predictions from the same raw input:

```bash
uv run python -m src.mlops verify-model \
  --model-name medical-insurance-cost \
  --alias champion \
  --output json
```

This check uses `mlflow.sklearn.load_model`, the flavor deployment must use for this pipeline.

### Rollback and token rotation

To roll back, review an earlier numeric version, confirm it already has complete lineage and is validated, then run `promote-model` with that earlier `--version`. This moves `champion`; resolve it again and pass the newly returned numeric URI to deployment. Existing deployments remain pinned until explicitly changed.

If a token is compromised, revoke it immediately in DagsHub user token settings, create a replacement, update `MLFLOW_TRACKING_PASSWORD` in local/CI secret storage, clear the old process environment, and rerun a read-only `inspect-model` or `resolve-model` command to confirm access. Do not alter the tracking URI or commit the replacement token.

Tracking settings:

- `ENABLE_MLFLOW_TRACKING` defaults to `false`.
- `MLFLOW_TRACKING_URI` is optional; enabled tracking defaults to the local `./mlruns` store.
- `MLFLOW_EXPERIMENT_NAME` defaults to `medical-insurance-cost`.
- `ENABLE_MODEL_REGISTRATION` defaults to `false` and requires tracking to be enabled.
- `MLFLOW_REGISTERED_MODEL_NAME` defaults to `medical-insurance-cost`.
- `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are required for a DagsHub URI.

## Immutable Modal deployment

DagsHub is a deployment-time dependency only. The packaging command resolves no aliases: it accepts exactly `models:/medical-insurance-cost/<positive-integer>`, validates registry and source-run lineage, downloads that version, verifies its serialized pipeline checksum, and creates `build/model/`. Modal then bakes that local package into the image. Container startup and predictions use only `/app/build/model`; no DagsHub credential is attached to the function.

Prediction persistence uses a separate Modal secret named
`medical-insurance-database`. After loading the Neon `DATABASE_URL` locally,
create or update it without placing its value in source:

```bash
uv run --all-extras --env-file .env bash -c '
  test -n "$DATABASE_URL" || { echo "DATABASE_URL is missing"; exit 1; }
  modal secret create --force medical-insurance-database \
    DATABASE_URL="$DATABASE_URL"
'
```

Only this database secret is attached to the serving function. MLflow and DagsHub
credentials remain deployment-time-only and are never attached to inference.

The production image uses Python 3.12 and `requirements-serving.txt`. It includes the API, schemas, prediction service, local runtime validator, and packaged model. It excludes frontend assets, datasets, notebooks, training code, registry and promotion code, tests, credentials, caches, and local artifacts.

### Authenticate Modal

Install the locked deployment extra and authenticate the local Modal CLI:

```bash
uv sync --locked --extra deployment
uv run modal setup
```

For GitHub Actions, create Modal token credentials and store them as `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. The protected `production` GitHub environment must contain these secrets plus:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

The deployment workflow has only `contents: read` repository permission. MLflow credentials are scoped to the alias-resolution and packaging steps; the Modal deployment step receives only Modal credentials.

### Prepare and validate a package manually

Load the DagsHub settings described above, resolve `champion` once, and record every returned identity value:

```bash
uv run python -m src.mlops resolve-model \
  --model-name medical-insurance-cost \
  --alias champion \
  --output json
```

Use the returned numeric URI, run ID, and checksum without resolving the alias again:

```bash
uv run python -m src.mlops prepare-deployment \
  --model-uri models:/medical-insurance-cost/7 \
  --output-dir build/model \
  --expected-run-id "<run-id>" \
  --expected-pipeline-sha256 "<sha256>" \
  --output json
```

The output contains `build/model/model/` and `build/model/deployment_metadata.json`. The command refuses aliases, stages, `latest`, run URIs, filesystem paths, zero or negative versions, incomplete metadata, unexpected files in the output directory, checksum mismatches, invalid signatures or examples, unsupported flavors, and failed or non-finite smoke predictions.

Revalidate the completed package without registry credentials or network access:

```bash
env -u MLFLOW_TRACKING_URI \
    -u MLFLOW_TRACKING_USERNAME \
    -u MLFLOW_TRACKING_PASSWORD \
  uv run python -m src.mlops validate-deployment \
    --package-dir build/model \
    --output json
```

### Serve and deploy

Both commands require a completed `build/model` package:

```bash
uv run modal serve modal_app.py
uv run modal deploy modal_app.py
```

The CLI prints the generated web endpoint. It is also visible on the Modal dashboard under the `medical-insurance-cost` app. After a deployment to another workspace, update the two production URLs near the top of this README.

For production, open **Actions → Deploy immutable model to Modal → Run workflow**, leave the alias as `champion`, and dispatch it against the protected `production` environment. The workflow resolves the alias exactly once, pins the numeric URI as a step output, validates and packages that exact version, deploys it, and records the version, run ID, source commit, dataset checksum, pipeline checksum, and deployment ID in the workflow summary.

Verify the deployed application:

```bash
APP_URL="https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run"

curl "$APP_URL/health"
curl "$APP_URL/docs"
curl -X POST "$APP_URL/predict-json" \
  -H "Content-Type: application/json" \
  -d '{"age":29,"sex":"female","bmi":27.4,"children":2,"smoker":"no","region":"southeast"}'
```

`/health` must return `{"status":"ok"}`. The deployment workflow summary and startup log fields `deployment_id`, `model_name`, `model_version`, `mlflow_run_id`, `source_commit_sha`, and `pipeline_sha256` identify exactly what is running.

### Roll back

Every rollback passes the same lineage, signature, input-example, checksum, loading, and smoke-prediction gates as a forward deployment. Never use `latest`.

There are two supported approaches:

1. Redeploy a previously recorded exact numeric URI using its recorded run ID and pipeline checksum.
2. Move `champion` to a previously validated numeric version with `promote-model`, dispatch the workflow, and let that new run resolve the alias once before packaging the returned numeric URI.

Moving `champion` after a workflow has resolved it cannot change that active deployment because every later step uses the recorded numeric URI.

## Run the API

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Open the interactive API documentation at <http://localhost:8000/docs>. FastAPI
is API-only; the user interface is served by Vite during development.

Health check:

```bash
curl http://localhost:8000/health
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

## Run the frontend

Start the backend as shown above, then use a second terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. In development, the frontend always sends a
relative request to `/predict-json`; Vite proxies that route to
`http://localhost:8000`, so local CORS workarounds and hard-coded URLs are not
needed.

Production is deployed by the Vercel project `medical-insurance-cost`, connected
to this GitHub repository with these settings:

| Setting | Value |
| --- | --- |
| Root Directory | `frontend` |
| Framework Preset | Vite |
| Install Command | `npm ci` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Node.js Version | `24.x` |

The Production and Preview environments define this public build-time variable:

```dotenv
VITE_API_BASE_URL=https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run
```

`VITE_*` values are embedded in browser JavaScript during `npm run build`. They
are public configuration and must never contain tokens, passwords, or other
secrets. Redeploy after changing `VITE_API_BASE_URL`; an existing build does not
pick up later environment changes. The frontend removes trailing slashes and
appends `/predict-json`. Production builds show a controlled configuration error
if this value is missing, malformed, contains credentials, or does not use
HTTP(S). The HTTPS Vercel frontend uses the HTTPS FastAPI endpoint to avoid
mixed-content blocking.

### Backend CORS allowlist

FastAPI reads a comma-separated list of exact origins. Local and production
origins can be enabled independently:

```dotenv
CORS_ALLOWED_ORIGINS=http://localhost:5173,https://medical-insurance-cost.vercel.app
```

Wildcard origins and origins containing credentials, paths, query strings, or
fragments are rejected. `modal_app.py` injects this value into the serving
function, and changing the allowlist requires redeploying Modal. Generated Vercel
preview URLs are intentionally not covered, so Preview builds render but their
prediction requests are blocked by CORS. Enable a live Preview prediction only
by allowlisting that trusted Preview origin exactly and redeploying Modal; do not
use a wildcard or a generated-domain pattern.

The production architecture is the Vercel-hosted `frontend/` calling the
separately deployed FastAPI backend. Vercel project IDs, account metadata,
`.vercel/`, local environment files, tokens, and credentials are not committed.

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

Backend checks:

```bash
uv run --extra dev --extra monitoring pytest
uv run --extra dev --extra monitoring ruff check .
```

Frontend checks:

```bash
cd frontend
npm run lint
npm test
npm run build
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
frontend/
├── src/
│   ├── components/
│   ├── services/
│   ├── types/
│   └── utils/
├── package.json
├── package-lock.json
└── vite.config.ts

src/
├── api/
│   ├── dependencies.py
│   └── routes/
├── mlops/
│   ├── config.py
│   ├── registry.py
│   └── tracking.py
├── monitoring/
│   ├── arize_client.py
│   ├── baseline.py
│   ├── exporter.py
│   ├── ground_truth.py
│   └── outbox.py
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
