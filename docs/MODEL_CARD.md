# Model Card: Medical Insurance Cost Prediction

## Model Details
- Problem type: Regression
- Model: Scikit-learn regressor (see training pipeline for exact estimator)
- Target: Medical insurance charges (currency in USD)

## Intended Use
- Estimate insurance charges from user-provided demographic and health-related inputs.
- For demo/educational use only; not for clinical or pricing decisions.

## Data
- Source: `Data/medical_insurance.csv`
- Features used:
  - age (int)
  - sex (categorical)
  - bmi (float)
  - children (int)
  - smoker (categorical)
  - region (categorical)
- Target: `charges` (float)

## Metrics
- Primary: R2 (reported by training pipeline)
- Note: Metrics are reported on the train/test split from the ingestion step.

## Limitations
- Small, static dataset with limited geographic and demographic coverage.
- Correlations may not generalize to other populations or policy structures.
- Sensitive attributes (e.g., sex) are included and may reflect historical bias.

## Fairness and Bias Considerations
- No formal fairness analysis was performed.
- The model may encode biases present in the dataset.

## Ethical Considerations
- Predictions are estimates; use with caution.
- Not intended for real-world pricing or eligibility decisions.

## How to Improve
- Collect more diverse and recent data.
- Add formal bias/fairness evaluation.
- Provide uncertainty estimates.