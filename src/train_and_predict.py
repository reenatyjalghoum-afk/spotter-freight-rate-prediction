import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor


# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(DATA_DIR / "train-test.csv")
validation = pd.read_csv(DATA_DIR / "validation.csv")
december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")


# ============================================================
# Data cleaning
# ============================================================

train["weight"] = train["weight"].abs()
validation["weight"] = validation["weight"].abs()
december["weight"] = december["weight"].abs()


# ============================================================
# Feature engineering
# ============================================================

def add_date_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


train = add_date_features(train)
validation = add_date_features(validation)
december = add_date_features(december)


# ============================================================
# Model features
# ============================================================

features = [
    "pickup",
    "delivery",
    "distance",
    "equipment",
    "weight",
    "month",
    "day_of_year",
    "day_of_week",
]

categorical_features = [
    "pickup",
    "delivery",
    "equipment",
]


# ============================================================
# Train final model
# ============================================================

X_train = train[features]
y_train = train["posted_rate"]

model = CatBoostRegressor(
    iterations=600,
    learning_rate=0.05,
    depth=7,
    loss_function="MAE",
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
)


# ============================================================
# Validation predictions
# ============================================================

X_validation = validation[features]

validation_predictions = model.predict(X_validation)

validation_predictions = np.nan_to_num(
    validation_predictions,
    nan=1.0,
    posinf=1.0,
    neginf=1.0,
)

validation_predictions = np.clip(
    validation_predictions,
    1e-6,
    None,
)

validation_output = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": validation_predictions,
})

validation_output.to_csv(
    PROJECT_DIR / "validation_predictions.csv",
    index=False,
)


# ============================================================
# December predictions
# ============================================================

X_december = december[features]

december_predictions = model.predict(X_december)

december_predictions = np.nan_to_num(
    december_predictions,
    nan=1.0,
    posinf=1.0,
    neginf=1.0,
)

december_predictions = np.clip(
    december_predictions,
    1e-6,
    None,
)

december_output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
    ]
].copy()

december_output["date"] = (
    december_output["date"].dt.strftime("%Y-%m-%d")
)

december_output["predicted_rate"] = december_predictions

december_output.to_csv(
    PROJECT_DIR / "december_predictions.csv",
    index=False,
)


# ============================================================
# Final checks
# ============================================================

print()
print("Pipeline completed successfully.")
print(f"Validation predictions: {len(validation_output):,}")
print(f"December predictions: {len(december_output)}")
print(
    f"Validation prediction range: "
    f"{validation_output['predicted_rate'].min():.2f} - "
    f"{validation_output['predicted_rate'].max():.2f}"
)
print(
    f"December prediction range: "
    f"{december_output['predicted_rate'].min():.2f} - "
    f"{december_output['predicted_rate'].max():.2f}"
)
