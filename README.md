# Spotter Freight Rate Prediction

Machine learning solution for the Spotter freight-rate prediction take-home assessment.

## Project Overview

This project predicts freight load rates using historical Spotter load data.

The solution includes:
- Data quality checks and cleaning
- Exploratory analysis
- Temporal validation
- Rate-per-mile baseline models
- CatBoost regression
- Predictions for 12,000 validation loads
- December 2025 fixed-input predictions
- Official scorer validation
- December prediction chart

## Dataset

The labeled development dataset contains **48,000 historical loads**.

The final validation set contains **12,000 loads**.

The December scenario contains **31 fixed-input rows**, where only the date changes.

## Data Cleaning

The `weight` feature contained missing values and negative values.

Negative weights were treated as sign errors and converted using:

`weight = abs(weight)`

Missing weights were retained because CatBoost can handle missing numerical values.

## Validation Strategy

A temporal validation strategy was used to better represent future prediction.

### Main Validation Split

- Training: January 1, 2025 – September 30, 2025
- Validation: October 1, 2025 – October 31, 2025

Rolling temporal validation was also evaluated on July, August, September, and October.

## Baseline

An equipment-specific median rate-per-mile baseline achieved:

**October MAE: 231.30**

## Final Model

The final model is a **CatBoost regression model**.

### Features

- pickup
- delivery
- distance
- equipment
- weight
- month
- day_of_year
- day_of_week

### Categorical Features

- pickup
- delivery
- equipment

### Model Configuration

| Parameter | Value |
|---|---:|
| Iterations | 600 |
| Learning rate | 0.05 |
| Depth | 7 |
| Loss function | MAE |
| Random seed | 42 |

## Validation Results

Rolling temporal validation results:

| Validation Month | MAE |
|---|---:|
| July | 148.15 |
| August | 90.43 |
| September | 108.49 |
| October | 115.41 |
| **Mean** | **115.62** |

The October direct CatBoost model achieved an MAE of **115.41** during model selection.

## Final Predictions

### Validation Predictions

`validation_predictions.csv`

Contains:
- `load_id`
- `predicted_rate`

The file contains exactly **12,000 predictions**.

### December Predictions

`december_predictions.csv`

Contains:
- `pickup`
- `delivery`
- `distance`
- `equipment`
- `weight`
- `date`
- `predicted_rate`

The file contains one prediction for each day from **December 1 through December 31, 2025**.

## Scoring

The provided scorer validates both prediction files and generates the December prediction chart.

Scorer location:

`src/SCORY/score.py`

The scorer successfully validated:
- **12,000 validation predictions**
- **31 December predictions**

Official final validation metrics are calculated by Spotter after submission.

## Installation

Install the required dependencies with:

`pip install -r requirements.txt`

## Reproducing the Predictions

The main training and prediction script is:

`src/train_and_predict.py`

It:
1. Loads the labeled development data.
2. Cleans the weight feature.
3. Creates date-based features.
4. Trains the CatBoost model.
5. Generates validation predictions.
6. Generates December predictions.
7. Saves the prediction files.

## Project Structure

```text
Spotter_Freight_Rate/
├── README.md
├── requirements.txt
├── validation_predictions.csv
├── december_predictions.csv
├── src/
│   ├── train_and_predict.py
│   └── SCORY/
│       └── score.py
└── reports/
    ├── Spotter_Freight_Rate_Assessment_Report_Professional.pdf
    └── spotter_report.md
```

## Report

The full technical report is available at:

`reports/Spotter_Freight_Rate_Assessment_Report_Professional.pdf`