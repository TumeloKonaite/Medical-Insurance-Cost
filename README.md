# Medical Insurance Cost Prediction

![CI](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml/badge.svg?branch=main&event=push)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/github/license/TumeloKonaite/Medical-Insurance-Cost)

Predict medical insurance charges using regression models.

## Live demo

**URL:** http://medical-insurance-cost-env.eba-pswdedzm.us-east-1.elasticbeanstalk.com/

![Demo preview](docs/demo.png)

**Try it (example inputs):**

- age: 29
- sex: female
- bmi: 27.4
- children: 2
- smoker: no
- region: southeast

**Expected output format:**

- `Estimated insurance charges: <number>`

**API docs:** http://medical-insurance-cost-env.eba-pswdedzm.us-east-1.elasticbeanstalk.com/docs

**Try it (curl):**

```bash
curl -X POST http://medical-insurance-cost-env.eba-pswdedzm.us-east-1.elasticbeanstalk.com/predict \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "age=29" \
  -d "sex=female" \
  -d "bmi=27.4" \
  -d "children=2" \
  -d "smoker=no" \
  -d "region=southeast"
```

## Demo highlights

- End-to-end pipeline from raw CSV to trained model and predictions.
- FastAPI web UI for recruiter-friendly testing.
- CI checks (Ruff + pytest) with reproducible artifacts.

**Quick run (one command):**

```powershell
uv run python scripts/run_pipeline.py
```

**Architecture overview:** see `docs/diagram.md`.

## Project structure

- `Data/medical_insurance.csv`: dataset used for modeling
- `notebooks/`: analysis workflow
- `src/`: reusable features and model modules

## Requirements

- Python 3.12+
- Common ML stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Dev tools (ruff + pytest):

```powershell
python -m pip install .[dev]
```

### Run the full pipeline

```powershell
uv run python scripts/run_pipeline.py
```
### Scripts

Minimal pipeline entry point:

```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

train_path, test_path, _ = DataIngestion().initiate_data_ingestion()
train_arr, test_arr, _ = DataTransformation().initiate_data_transformation(
    train_path, test_path
)
score = ModelTrainer().initiate_model_trainer(train_arr, test_arr)
print("Model score (R2):", score)
```

## Notes

- `split_data` uses a default `test_size=0.7` (30/70 train/test).
- Demo runs in single-instance mode to save cost.
- HA configs are available (autoscaling + rolling updates) in `deploy/ha/`.

To enable HA:

```powershell
Copy-Item deploy\ha\*.config .ebextensions\
```

## License

MIT License. See `LICENSE`.
