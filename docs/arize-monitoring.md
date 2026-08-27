# Neon outbox and Arize monitoring

[Back to the project README](../README.md)

This runbook covers prediction-event persistence, the transactional export
outbox, Arize validation and production uploads, delayed ground truth, monitor
configuration, backlog inspection, and credential rotation.

## Data flow

```text
POST /predict-json
    → run inference
    → commit prediction event + pending outbox event in Neon
    → return without contacting Arize

Hourly Modal exporter
    → claim pending or stale outbox records
    → group by exact model version and event type
    → batch-upload to Arize
    → mark acknowledged rows as sent
    → reschedule failures with bounded backoff
```

Neon remains the durable source of truth. Arize delivery is asynchronous and
at-least-once; every retry reuses `request_id` as the stable Arize prediction ID.
Persistence is fail-open for the prediction response, while the exporter fails
closed on missing configuration or an upload failure.

## Stored prediction fields

Successful requests store typed columns for:

| Category | Fields |
| --- | --- |
| Identity and time | `id`, `request_id`, `created_at`, `source` |
| Model features | `age`, `sex`, `bmi`, `children`, `smoker`, `region` |
| Prediction metadata | `predicted_charges`, `model_version`, `prediction_contract_version`, `inference_latency_ms` |
| Delayed labels | `actual_charges`, `actual_recorded_at` |

Invalid requests and failed inference attempts are not stored. Duplicate request
IDs do not create duplicate prediction or outbox records.

## Configure Neon and migrate

Use a pooled Neon PostgreSQL URL in an ignored `.env` file:

```dotenv
DATABASE_URL=postgresql://<role>:<password>@<endpoint>-pooler.<region>.aws.neon.tech/<database>?sslmode=require
```

Apply the prediction and outbox migrations to the intended database:

```bash
uv run --env-file .env alembic upgrade head
```

The outbox migration also queues existing prediction events once. New prediction
and outbox rows are committed atomically.

Create or replace the Modal database secret:

```bash
uv run --all-extras --env-file .env bash -c '
  test -n "$DATABASE_URL" || { echo "DATABASE_URL is missing"; exit 1; }
  modal secret create --force medical-insurance-database \
    DATABASE_URL="$DATABASE_URL"
'
```

## Configure Arize AX

Create an [Arize AX account](https://arize.com/), create a space, then place the
API key and space ID in the ignored `.env` file:

```dotenv
ARIZE_API_KEY=<arize-api-key>
ARIZE_SPACE_ID=<arize-space-id>
ARIZE_MODEL_NAME=medical-insurance-cost
ARIZE_EXPORT_BATCH_SIZE=500
```

Create the dedicated Modal exporter secret:

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

This secret is attached only to the exporter. Missing Arize configuration never
prevents FastAPI from starting or returning predictions.

## Upload the validation baseline

Build the immutable production package before uploading its held-out test data:

```bash
uv sync --locked --extra monitoring
uv run --extra monitoring --env-file .env python -m src.monitoring upload-baseline \
  --test-data artifacts/test.csv \
  --model-package build/model
```

The command validates the packaged model, six-field signature,
`deployment_metadata.json`, exact dataset columns, targets, and generated
predictions. It uploads stable row IDs, raw features, predictions, actuals, the
exact numeric model version, and MLflow run ID to the Arize validation
environment. Baseline upload is an explicit release operation and never runs at
API startup.

## Deploy and run the exporter

Deploying the Modal application activates the cron schedule at minute 5 of every
hour:

```bash
uv sync --locked --extra deployment
uv run modal deploy modal_app.py
```

Run the production function manually after setup or during an operational check:

```bash
uv run modal run modal_app.py::export_predictions_to_arize
```

Run a one-shot exporter locally:

```bash
uv run --extra monitoring --env-file .env python -m src.monitoring export
```

Each run logs a sanitized summary containing claimed, sent, retried, failed, and
remaining records plus the oldest pending age. Failed batches are returned to
`pending` with bounded exponential backoff. Records abandoned in `processing`
are recovered after `ARIZE_CLAIM_STALE_MINUTES`, which defaults to 30.

## Record delayed ground truth

There is intentionally no public actual-label endpoint. An authorized operator
can record a charge by the stored request ID:

```bash
uv run --env-file .env python -m src.monitoring record-actual \
  --request-id <uuid> \
  --actual-charges 12345.67
```

The command requires a finite non-negative value, updates the prediction, and
creates an `actual` outbox event in one transaction. Repeating the same value is
idempotent; attempting to overwrite it with a different value is rejected.

## Inspect operations

Open the [medical-insurance-cost model directly in Arize AX][arize-model], then
use its model-level **Monitor** and **Dashboard** tabs.

[arize-model]: https://app.arize.com/organizations/QWNjb3VudE9yZ2FuaXphdGlvbjo0OTE3OTo2RjBo/spaces/U3BhY2U6NTI3MzA6SWZIcw==/models/modelName/medical-insurance-cost?selectedTab=performance

Inspect scheduled-function logs:

```bash
uv run modal app logs medical-insurance-cost --since 24h --timestamps
```

Inspect the backlog without selecting features or charges:

```sql
SELECT status, event_type, count(*) AS records, min(created_at) AS oldest
FROM arize_export_events
WHERE status <> 'sent'
GROUP BY status, event_type;
```

## Recommended monitors

For this low-volume application, use daily or multi-day evaluation windows and a
reasonable ingestion delay instead of noisy hourly statistical alerts.

- Prediction count and no-data
- Prediction-distribution drift
- Drift on the most influential features
- Numeric feature range and data quality
- Unexpected categorical values
- p95 inference latency
- Traffic grouped by model version
- MAE, RMSE, and R² after actual labels arrive
- Actual-label coverage and delay, where supported by a custom metric

Arize Free retention and usage limits can change. Keep the complete history in
Neon and check the current [Arize pricing](https://arize.com/pricing/).

## Rotate the Arize key

Create a replacement key, overwrite the Modal secret, redeploy, run one manual
export, and only then revoke the old key:

```bash
uv run modal secret create --force medical-insurance-arize \
  ARIZE_API_KEY="$NEW_ARIZE_API_KEY" \
  ARIZE_SPACE_ID="$ARIZE_SPACE_ID"
uv run modal deploy modal_app.py
uv run modal run modal_app.py::export_predictions_to_arize
```

## Privacy safeguards

- Never log or commit Arize credentials or database URLs.
- Do not export names, email addresses, IP addresses, database row IDs, raw
  bodies, free-form text, SHAP values, traces, or training candidates.
- Export only the six approved features, prediction and actual values, stable
  prediction ID, exact model version, source, contract version, and latency.
- These fields can still be sensitive. This integration is approved only for the
  repository's synthetic/demo data.
- CI uses fake clients and never contacts Arize or requires real credentials.
