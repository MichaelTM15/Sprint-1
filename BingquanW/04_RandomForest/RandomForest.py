
# ============================================================
# Random Forest - Complete-case Analysis
# Football Match Outcome Classification
#
# Target:
# FTR = H (Home Win), D (Draw), A (Away Win)
#
# Predictors:
# Full-time match statistics (excluding goals and betting odds)
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. FILE PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent

TRAIN_PATH = (
    ROOT / "processed-data" / "football_train_data.csv"
)

TEST_PATH = (
    ROOT / "processed-data" / "football_test_data.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n===== LOADING DATA =====")

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
CLASSES = ["H", "D", "A"]

# Check required columns
required_columns = FEATURES + [TARGET]

for name, dataset in [
    ("Training", train),
    ("Testing", test)
]:
    missing_columns = [
        col for col in required_columns
        if col not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} dataset is missing columns: "
            f"{missing_columns}"
        )

# Keep valid target labels
train = train[
    train[TARGET].isin(CLASSES)
].copy()

test = test[
    test[TARGET].isin(CLASSES)
].copy()

X_train = train[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = test[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

y_train = train[TARGET].copy()
y_test = test[TARGET].copy()

print("Original training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Number of features:", len(FEATURES))


# ============================================================
# 3. COMPLETE-CASE ANALYSIS
# ============================================================

print("\n===== COMPLETE-CASE ANALYSIS =====")

print("\nMissing values in training features:")
print(X_train.isna().sum())

# Identify training rows with missing predictors
missing_mask = X_train.isna().any(axis=1)

removed_rows = int(missing_mask.sum())

print("\nRows with missing predictors:", removed_rows)

print(
    "Indices of removed rows:",
    X_train.index[missing_mask].tolist()
)

# Remove incomplete rows from training data only
complete_mask = ~missing_mask

X_train_clean = X_train.loc[
    complete_mask
].copy()

y_train_clean = y_train.loc[
    complete_mask
].copy()

print(
    "Training rows after cleaning:",
    len(X_train_clean)
)

# Do not silently remove test observations
test_missing = X_test.isna().sum()

if test_missing.sum() > 0:
    raise ValueError(
        "The test dataset contains missing predictors. "
        "Review test preprocessing before evaluation.\n"
        + str(test_missing[test_missing > 0])
    )

# Save a summary of cleaning
cleaning_summary = pd.DataFrame({
    "Measure": [
        "Original training rows",
        "Removed training rows",
        "Remaining training rows",
        "Testing rows"
    ],
    "Value": [
        len(X_train),
        removed_rows,
        len(X_train_clean),
        len(X_test)
    ]
})

cleaning_summary.to_csv(
    OUTPUT / "rf_complete_case_cleaning.csv",
    index=False
)


# ============================================================
# 4. TRAIN RANDOM FOREST
# ============================================================

print("\n===== TRAINING RANDOM FOREST =====")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_clean,
    y_train_clean
)

print("Random Forest training completed.")


# ============================================================
# 5. MAKE PREDICTIONS
# ============================================================

print("\n===== MAKING PREDICTIONS =====")

y_pred = model.predict(X_test)

probabilities = model.predict_proba(X_test)

probability_df = pd.DataFrame(
    probabilities,
    columns=[
        f"Probability_{label}"
        for label in model.classes_
    ],
    index=X_test.index
)

predictions_df = pd.DataFrame({
    "Actual_FTR": y_test,
    "Predicted_FTR": y_pred
}, index=X_test.index)

for label in CLASSES:
    predictions_df[
        f"Probability_{label}"
    ] = probability_df[
        f"Probability_{label}"
    ]

predictions_df.to_csv(
    OUTPUT / "rf_complete_case_predictions.csv",
    index=False
)

print("Predictions saved.")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

print("\n===== MODEL EVALUATION =====")

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

draw_recall = recall_score(
    y_test,
    y_pred,
    labels=["D"],
    average="macro",
    zero_division=0
)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Draw Recall: {draw_recall:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0
    )
)

metrics_df = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Macro_F1",
        "Draw_Recall"
    ],
    "Value": [
        accuracy,
        macro_f1,
        draw_recall
    ]
})

metrics_df.to_csv(
    OUTPUT / "rf_complete_case_metrics.csv",
    index=False
)


# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=CLASSES
)

print(
    pd.DataFrame(
        cm,
        index=CLASSES,
        columns=CLASSES
    )
)

fig, ax = plt.subplots(figsize=(7, 6))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Home Win",
        "Draw",
        "Away Win"
    ]
)

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(
    "Random Forest - Complete-case Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "rf_complete_case_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

print("\n===== FEATURE IMPORTANCE =====")

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
)

print(importance_df.to_string(index=False))

importance_df.to_csv(
    OUTPUT / "rf_complete_case_feature_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(9, 6))

plot_data = importance_df.sort_values(
    "Importance",
    ascending=True
)

ax.barh(
    plot_data["Feature"],
    plot_data["Importance"]
)

ax.set_xlabel("Feature Importance")

ax.set_title(
    "Random Forest - Complete-case Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT / "rf_complete_case_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n======================================")
print("FINAL RANDOM FOREST RESULTS")
print("======================================")

print("Missing-data method: Complete-case")
print("Removed training rows:", removed_rows)
print("Remaining training rows:", len(X_train_clean))

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Draw Recall: {draw_recall * 100:.2f}%")

print("\nAnalysis completed successfully!")
print("Results saved in:", OUTPUT)
