# Spotter Freight Rate Prediction Assessment

## 1. Executive Summary

This project develops a machine learning model for freight load rate prediction using the provided Spotter development dataset.

The development dataset contains 48,000 labeled loads, while the final validation set contains 12,000 loads.

A temporal validation strategy was used to evaluate future prediction performance. The final model is a CatBoost regression model using route, equipment, distance, weight, and calendar features.

The final prediction files were successfully validated using the provided scoring script.

## 2. Data Overview

The development data contains 48,000 historical freight loads.

The validation set contains 12,000 loads.

The December scenario contains 31 rows from December 1 through December 31, 2025, with fixed route and load characteristics.

## 3. Data Cleaning

The weight feature contained missing values and negative values.

Negative weights were treated as sign errors and converted using:

weight = abs(weight)

Missing weight values were retained because CatBoost can handle missing numerical values.

## 4. Exploratory Analysis

Distance showed a strong relationship with posted rate, with a correlation of approximately 0.909 in the cleaned training data.

Equipment type also showed differences in average rate and rate-per-mile.

## 5. Baseline

The overall median rate-per-mile baseline achieved an October MAE of 253.85.

An equipment-specific median rate-per-mile baseline achieved an October MAE of 231.30.

## 6. Validation Strategy

A temporal validation strategy was used.

Primary split:

- Training: January 1, 2025 through September 30, 2025
- Validation: October 1, 2025 through October 31, 2025

Rolling temporal validation was also performed using July, August, September, and October as validation months.

## 7. Model

CatBoost regression was selected because it supports categorical features directly and handles missing numerical values.

Features:

- pickup
- delivery
- distance
- equipment
- weight
- month
- day_of_year
- day_of_week

Categorical features:

- pickup
- delivery
- equipment

Final configuration:

- iterations: 600
- learning_rate: 0.05
- depth: 7
- loss_function: MAE
- random_seed: 42

## 8. Validation Results

| Validation Month | MAE |
|---|---:|
| July | 148.15 |
| August | 90.43 |
| September | 108.49 |
| October | 115.41 |
| Mean | 115.62 |

The October direct CatBoost model achieved an MAE of 115.41 during model selection.

## 9. Final Predictions

The final model was trained using all 48,000 cleaned training rows.

It generated:

- 12,000 validation predictions
- 31 December predictions

The validation file contains:

load_id,predicted_rate

The December file contains:

pickup,delivery,distance,equipment,weight,date,predicted_rate

## 10. December 2025 Scenario

Fixed inputs:

- Pickup: Lexington
- Delivery: Fort Wayne
- Distance: 360 miles
- Equipment: Dry Van
- Weight: 32,000 lb

Only the date changes across the 31 December rows.

The predicted rates are approximately between $831 and $851.

The provided scorer generated:

scorer_results/candidate_december.png

## 11. Submission Validation

The provided scoring script successfully validated:

- 12,000 final validation predictions
- 31 December predictions
- Validation IDs
- Positive prediction values
- December dates
- Fixed December inputs

The December chart was generated successfully.

Official final validation metrics are calculated by Spotter after submission.

## 12. Reproducibility

The project includes:

- requirements.txt
- README.md
- src/train_and_predict.py
- score.py
- validation_predictions.csv
- december_predictions.csv
- December prediction chart

## 13. Conclusion

The project provides a complete freight-rate prediction pipeline covering data cleaning, exploratory analysis, temporal validation, model development, final prediction generation, and scorer validation.

The selected CatBoost model improved over the equipment-specific rate-per-mile baseline on the October temporal validation split.

All required prediction files passed the provided scoring checks.
