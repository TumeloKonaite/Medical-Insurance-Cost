# Medical Insurance Cost Prediction
[![CI](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml/badge.svg)](https://github.com/TumeloKonaite/Medical-Insurance-Cost/actions/workflows/ci.yml)

Predict medical insurance charges using regression models, with baseline training, random-forest tuning, and error analysis.

## Project structure

- `Data/medical_insurance.csv`: dataset used for modeling
- `notebooks/`: analysis workflow (EDA; feature engineering; modeling; tuning; error analysis)
- `main.py`: placeholder entry point

## Requirements

- Python 3.12+
- Common ML stack: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

Install dependencies (example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install numpy pandas scikit-learn matplotlib seaborn
```



### Scripts (module-style)

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

### Random forest tuning

```python
from src.models.tune_rf import tune_random_forest

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid = tune_random_forest(X_train_p, y_train, param_grid)
best_model = grid.best_estimator_
```

## Notes

- `split_data` uses a default `test_size=0.7`, which is a 30/70 train/test split. Adjust as needed.
- The preprocessing pipeline one-hot encodes categorical variables and MinMax-scales numeric features.

## License

Add a license if you plan to distribute this project.
