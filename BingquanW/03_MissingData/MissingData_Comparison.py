
# ============================================================
# Missing Data Handling Methods - Random Forest Comparison
#
# 1. Complete-case Analysis
# 2. Available-case Analysis
# 3a. Simple Imputation (Mean)
# 3b. Regression Imputation
# 3c. Multiple Imputation
#
# Research question:
# Can we predict Premier League match outcomes
# from full-time match statistics?
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report
)


# ============================================================
# 1. LOAD DATA
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent

TRAIN_PATH = (
    ROOT / "processed-data" / "football_train_data.csv"
)
TEST_PATH = (
    ROOT / "processed-data" / "football_test_data.csv"
)

train = pd.read_csv(TRAIN_PATH, low_memory=False)
test = pd.read_csv(TEST_PATH, low_memory=False)

FEATURES = [
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR"
]

TARGET = "FTR"
CLASSES = np.array(["A", "D", "H"])

# Keep valid match outcomes
train = train[
    train[TARGET].isin(CLASSES)
].copy()

test = test[
    test[TARGET].isin(CLASSES)
].copy()

# Convert predictors to numeric
X_train = train[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = test[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

y_train = train[TARGET]
y_test = test[TARGET]

print("\n===== DATASET INFORMATION =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Number of predictors:", len(FEATURES))


# ============================================================
# 2. IDENTIFY MISSING VALUES
# ============================================================

print("\n===== MISSING VALUE ANALYSIS =====")

missing_counts = X_train.isna().sum()

print("\nMissing values in each predictor:")
print(missing_counts)

print("\nTotal missing values:", missing_counts.sum())

missing_rows = X_train.isna().any(axis=1)

print("Number of problematic rows:", missing_rows.sum())
print(
    "Problematic row indices:",
    X_train.index[missing_rows].tolist()
)

# Save the original problematic observations
train.loc[missing_rows].to_csv(
    OUTPUT / "problematic_rows.csv",
    index=True
)

# Check the held-out test dataset
print("\nMissing values in test data:")
print(X_test.isna().sum())

if X_test.isna().any().any():
    raise ValueError(
        "Test predictors contain missing values. "
        "Additional test preprocessing is required."
    )


# ============================================================
# 3. RANDOM FOREST MODEL AND METRICS
# ============================================================

def create_random_forest():

    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )


results = []


def evaluate_model(method_name, predictions):

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    draw_recall = recall_score(
        y_test,
        predictions,
        labels=["D"],
        average="macro",
        zero_division=0
    )

    results.append({
        "Method": method_name,
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Draw_Recall": draw_recall
    })

    print(f"\n===== {method_name} =====")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Draw Recall: {draw_recall:.4f}")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["H", "D", "A"],
            zero_division=0
        )
    )


# ============================================================
# METHOD 1: COMPLETE-CASE ANALYSIS
# ============================================================

print("\nRunning Complete-case Analysis...")

# Remove training observations with missing predictors
complete_mask = ~missing_rows

X_train_complete = X_train.loc[complete_mask]
y_train_complete = y_train.loc[complete_mask]

print("Original training rows:", len(X_train))
print("Remaining rows:", len(X_train_complete))
print("Removed rows:", missing_rows.sum())

model_complete = create_random_forest()

model_complete.fit(
    X_train_complete,
    y_train_complete
)

pred_complete = model_complete.predict(X_test)

evaluate_model(
    "Complete-case Analysis",
    pred_complete
)


# ============================================================
# METHOD 2: AVAILABLE-CASE ANALYSIS
# ============================================================

print("\nRunning Available-case Analysis...")

# Each feature is analysed using all its observed values.
# Rows are not removed because of missingness in
# unrelated variables.

available_summary = pd.DataFrame({
    "Available_Count": X_train.count(),
    "Missing_Count": X_train.isna().sum(),
    "Mean": X_train.mean(),
    "Median": X_train.median(),
    "Standard_Deviation": X_train.std()
})

print("\nAvailable-case summary:")
print(available_summary.round(3))

available_summary.to_csv(
    OUTPUT / "available_case_summary.csv"
)

# Pairwise complete observations are used for
# each correlation calculation.
available_correlations = X_train.corr()

print("\nAvailable-case correlation matrix:")
print(available_correlations.round(3))

available_correlations.to_csv(
    OUTPUT / "available_case_correlations.csv"
)

# Available-case analysis is used for descriptive
# statistics here, not as a separate RF model.


# ============================================================
# METHOD 3a: SIMPLE IMPUTATION (MEAN)
# ============================================================

print("\nRunning Simple Imputation (Mean)...")

mean_imputer = SimpleImputer(
    strategy="mean",
    keep_empty_features=True
)

# Fit imputer only on training data
X_train_mean = mean_imputer.fit_transform(
    X_train
)

X_test_mean = mean_imputer.transform(
    X_test
)

model_mean = create_random_forest()

model_mean.fit(
    X_train_mean,
    y_train
)

pred_mean = model_mean.predict(
    X_test_mean
)

evaluate_model(
    "Simple Imputation (Mean)",
    pred_mean
)


# ============================================================
# METHOD 3b: REGRESSION IMPUTATION
# ============================================================

print("\nRunning Regression Imputation...")

regression_imputer = IterativeImputer(
    max_iter=10,
    sample_posterior=False,
    random_state=42
)

# Fit regression imputer on training data only
X_train_regression = regression_imputer.fit_transform(
    X_train
)

X_test_regression = regression_imputer.transform(
    X_test
)

model_regression = create_random_forest()

model_regression.fit(
    X_train_regression,
    y_train
)

pred_regression = model_regression.predict(
    X_test_regression
)

evaluate_model(
    "Regression Imputation",
    pred_regression
)


# ============================================================
# METHOD 3c: MULTIPLE IMPUTATION
# ============================================================

print("\nRunning Multiple Imputation...")

NUMBER_OF_IMPUTATIONS = 5

all_probabilities = []

for i in range(NUMBER_OF_IMPUTATIONS):

    print(
        f"Imputation {i + 1}/{NUMBER_OF_IMPUTATIONS}",
        flush=True
    )

    multiple_imputer = IterativeImputer(
        max_iter=10,
        sample_posterior=True,
        random_state=42 + i
    )

    X_train_multiple = multiple_imputer.fit_transform(
        X_train
    )

    X_test_multiple = multiple_imputer.transform(
        X_test
    )

    model_multiple = create_random_forest()

    model_multiple.fit(
        X_train_multiple,
        y_train
    )

    probabilities = model_multiple.predict_proba(
        X_test_multiple
    )

    # Ensure the same class order for all models
    aligned_probabilities = np.zeros(
        (len(X_test), len(CLASSES))
    )

    for j, label in enumerate(model_multiple.classes_):

        class_index = np.where(
            CLASSES == label
        )[0][0]

        aligned_probabilities[:, class_index] = (
            probabilities[:, j]
        )

    all_probabilities.append(
        aligned_probabilities
    )

# Combine predictions from the five models
average_probabilities = np.mean(
    all_probabilities,
    axis=0
)

pred_multiple = CLASSES[
    np.argmax(
        average_probabilities,
        axis=1
    )
]

evaluate_model(
    "Multiple Imputation",
    pred_multiple
)


# ============================================================
# 4. FINAL MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(results)

comparison["Accuracy_Percent"] = (
    comparison["Accuracy"] * 100
)

comparison["Macro_F1_Percent"] = (
    comparison["Macro_F1"] * 100
)

comparison["Draw_Recall_Percent"] = (
    comparison["Draw_Recall"] * 100
)

print("\n========================================")
print("FINAL RANDOM FOREST COMPARISON")
print("========================================")

print(
    comparison[
        [
            "Method",
            "Accuracy_Percent",
            "Macro_F1_Percent",
            "Draw_Recall_Percent"
        ]
    ].round(2).to_string(index=False)
)

comparison.to_csv(
    OUTPUT / "missing_data_comparison.csv",
    index=False
)

print("\nAnalysis completed successfully!")
print("Results saved to:", OUTPUT)

