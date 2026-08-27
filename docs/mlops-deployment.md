# MLflow registry and immutable deployment

[Back to the project README](../README.md)

This runbook covers remote experiment tracking in DagsHub, model registration and
promotion, immutable package creation, GitHub Actions deployment to Modal, and
rollback.

## Design guarantees

- Training compares multiple regressors and registers only the selected fitted
  preprocessing-and-model pipeline.
- Registry versions carry source, dataset, schema, run, and pipeline-checksum
  lineage.
- Promotion is explicit: a reviewed numeric version receives
  `validation_status=validated` and the `champion` alias.
- Deployment resolves `champion` exactly once and uses the returned immutable
  numeric URI for every subsequent step.
- The production image contains the validated package but no DagsHub credentials,
  training code, datasets, notebooks, or registry client.

## Configure DagsHub MLflow

Every DagsHub repository exposes an MLflow endpoint at
`https://dagshub.com/<owner>/<repository>.mlflow`. Copy `.env.example` to an
ignored `.env` and configure:

```dotenv
ENABLE_MLFLOW_TRACKING=true
ENABLE_MODEL_REGISTRATION=true
MLFLOW_TRACKING_URI=https://dagshub.com/<username>/<repository>.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-access-token>
MLFLOW_EXPERIMENT_NAME=medical-insurance-cost
MLFLOW_REGISTERED_MODEL_NAME=medical-insurance-cost
```

Create the token in the
[DagsHub token settings](https://dagshub.com/user/settings/tokens) and give the
account contributor access to the repository. Keep the token out of URLs, shell
history, logs, screenshots, commits, and issue text.

The configuration fails closed when remote credentials are supplied without an
explicit remote URI. Authentication and remote service errors are sanitized.

## Train and review a candidate

```bash
uv run --env-file .env python scripts/run_pipeline.py
```

The command records candidate and selected metrics, parameters, signature, input
example, Git commit, dataset checksum, and pipeline checksum. It registers the
selected model as the next numeric version of `medical-insurance-cost` without
assigning `champion`.

Review the run and model version in DagsHub. A candidate version contains:

- `training_run_id`
- `source_commit_sha`
- `dataset_sha256`
- `pipeline_sha256`
- `feature_schema_version`
- `prediction_contract_version`
- `selected_model`
- `selection_metric`
- `validation_status=candidate`

Inspect one exact version from the terminal:

```bash
uv run --env-file .env python -m src.mlops inspect-model \
  --model-name medical-insurance-cost \
  --version 7 \
  --output json
```

## Promote, resolve, and verify

After reviewing the candidate, promote it explicitly:

```bash
uv run --env-file .env python -m src.mlops promote-model \
  --model-name medical-insurance-cost \
  --version 7 \
  --alias champion \
  --confirm \
  --output json
```

Resolve the alias before deployment:

```bash
uv run --env-file .env python -m src.mlops resolve-model \
  --model-name medical-insurance-cost \
  --alias champion \
  --output json
```

The resolver accepts only a validated positive numeric version with complete
lineage and valid checksums. Its returned URI is immutable, for example
`models:/medical-insurance-cost/7`. Never deploy an alias, stage, `latest`, run
URI, or filesystem path.

Verify that alias and numeric loading identify the same run and produce compatible
finite predictions:

```bash
uv run --env-file .env python -m src.mlops verify-model \
  --model-name medical-insurance-cost \
  --alias champion \
  --output json
```

## Configure Modal

Install deployment dependencies and authenticate the CLI:

```bash
uv sync --locked --extra deployment
uv run modal setup
```

Create the prediction database secret from the ignored `.env` file:

```bash
uv run --all-extras --env-file .env bash -c '
  test -n "$DATABASE_URL" || { echo "DATABASE_URL is missing"; exit 1; }
  modal secret create --force medical-insurance-database \
    DATABASE_URL="$DATABASE_URL"
'
```

Only this database secret is attached to FastAPI. The separate Arize secret is
attached only to the scheduled exporter, and DagsHub credentials are never
attached to either production function.

## Prepare a package manually

Use the numeric URI, run ID, and pipeline checksum returned by the single alias
resolution:

```bash
uv run --env-file .env python -m src.mlops prepare-deployment \
  --model-uri models:/medical-insurance-cost/7 \
  --output-dir build/model \
  --expected-run-id "<run-id>" \
  --expected-pipeline-sha256 "<sha256>" \
  --output json
```

The output contains the MLflow model and `deployment_metadata.json`. Packaging
validates registry lineage, allowed files and flavors, the serialized checksum,
signature, input example, model load, and a finite smoke prediction.

Revalidate the finished package without registry credentials or network access:

```bash
env -u MLFLOW_TRACKING_URI \
    -u MLFLOW_TRACKING_USERNAME \
    -u MLFLOW_TRACKING_PASSWORD \
  uv run python -m src.mlops validate-deployment \
    --package-dir build/model \
    --output json
```

Deploy or serve the exact local package:

```bash
uv run modal serve modal_app.py
uv run modal deploy modal_app.py
```

## Deploy through GitHub Actions

The protected `production` GitHub environment requires:

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`
- `MLFLOW_TRACKING_URI`
- `MLFLOW_TRACKING_USERNAME`
- `MLFLOW_TRACKING_PASSWORD`

Open **Actions → Deploy immutable model to Modal → Run workflow**, retain the
`champion` alias, and dispatch the workflow. The workflow:

1. resolves the validated alias once;
2. pins the numeric URI and expected lineage as step outputs;
3. validates and packages that exact version;
4. deploys the baked image to Modal; and
5. records the deployment ID, model version, run ID, source commit, dataset
   checksum, and pipeline checksum in the workflow summary.

The workflow has only `contents: read` repository permission. MLflow credentials
are scoped to resolution and packaging; the Modal deployment step receives only
Modal credentials.

## Verify production

```bash
APP_URL="https://tumelokonaitedev--medical-insurance-cost-fastapi-app.modal.run"

curl "$APP_URL/health"
curl "$APP_URL/docs"
curl -X POST "$APP_URL/predict-json" \
  -H "Content-Type: application/json" \
  -d '{"age":29,"sex":"female","bmi":27.4,"children":2,"smoker":"no","region":"southeast"}'
```

`/health` must return `{"status":"ok"}`. The workflow summary and startup log
identify the exact deployment and model lineage.

## Rollback

Use either of these supported paths:

1. Redeploy a previously recorded exact numeric URI with its recorded run ID and
   pipeline checksum.
2. Validate and promote an earlier numeric version, move `champion`, and dispatch
   the deployment workflow again.

Every rollback passes the same lineage, checksum, package, loading, and smoke-test
gates as a forward deployment. Moving `champion` after a workflow has resolved it
cannot alter the active deployment.

## Rotate credentials

If a DagsHub token is compromised, revoke it, create a replacement, update the
local and GitHub secrets, clear the old environment, and run a read-only
`inspect-model` or `resolve-model` command. Do not change the tracking URI during
token rotation.
