
# ==========================================
# Random Forest - Football Match Prediction
# ==========================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. File paths
project_dir = Path(__file__).resolve().parent.parent
output_dir = Path(__file__).resolve().parent

train_path = project_dir / "processed-data" / "football_train_data.csv"
test_path = project_dir / "processed-data" / "football_test_data.csv"

# 2. Load datasets
train = pd.read_csv(train_path, low_memory=False)
test = pd.read_csv(test_path, low_memory=False)

print("Training data:", train.shape)
print("Testing data:", test.shape)

# 3. Select features
features = [
    "HS", "AS",
    "HST", "AST",
    "HF", "AF",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR"
]

target = "FTR"

# Check required columns
required_columns = features + [target]

for name, data in [("Train", train), ("Test", test)]:
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        raise ValueError(f"{name} data is missing columns: {missing}")

# 4. Prepare X and y
X_train = train[features].apply(pd.to_numeric, errors="coerce")
y_train = train[target]

X_test = test[features].apply(pd.to_numeric, errors="coerce")
y_test = test[target]

# Remove rows without valid target labels
train_valid = y_train.isin(["H", "D", "A"])
test_valid = y_test.isin(["H", "D", "A"])

X_train = X_train.loc[train_valid]
y_train = y_train.loc[train_valid]

X_test = X_test.loc[test_valid]
y_test = y_test.loc[test_valid]

# 5. Handle missing values
imputer = SimpleImputer(strategy="median", keep_empty_features=True)

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 6. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# 7. Train model
rf_model.fit(X_train, y_train)

# 8. Make predictions
y_pred = rf_model.predict(X_test)

# 9. Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n===== Random Forest Results =====")
print("Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["H", "D", "A"],
        zero_division=0
    )
)

# 10. Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["H", "D", "A"],
    display_labels=["Home Win", "Draw", "Away Win"],
    cmap="Blues"
)

plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(output_dir / "random_forest_confusion_matrix.png", dpi=300)
plt.close()

# 11. Feature Importance
importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values()

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.savefig(output_dir / "random_forest_feature_importance.png", dpi=300)
plt.close()

# 12. Actual vs Predicted Results
comparison = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred
})

comparison.to_csv(
    output_dir / "random_forest_predictions.csv",
    index=False
)

print("\nResults saved successfully!")
