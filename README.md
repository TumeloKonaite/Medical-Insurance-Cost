# Medical Insurance Cost Prediction

![CI](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml/badge.svg?branch=main&event=push)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/github/license/TumeloKonaite/Medical-Insurance-Cost)

Predict medical insurance charges using regression models.

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
python -m pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the full pipeline

```powershell
uv run python scripts/run_pipeline.py
```
### Scripts

In your own script or notebook:

```python
import pandas as pd
from src.features.build_features import get_feature_groups, make_preprocessor, split_data, fit_transform
from src.models.train_baselines import train_baselines

# Load data
_df = pd.read_csv('Data/medical_insurance.csv')

# Feature prep
cat_cols, int_cols, float_cols, num_cols = get_feature_groups(_df)
X_train, X_test, y_train, y_test = split_data(_df, target_column='charges')
preprocessor = make_preprocessor(cat_cols, num_cols)
X_train_p, X_test_p = fit_transform(preprocessor, X_train, X_test)

# Train baselines
train_baselines(X_train_p, y_train, X_test_p, y_test)
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
