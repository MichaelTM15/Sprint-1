# Exploratory Data Analysis
# This section explores relationships between full-time match statistics
# and match outcomes in English football.
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# ==============================================
# Exploratory Data Analysis
# Shots on Target Difference vs Match Result
# ==============================================

# 1. Load the half-cleaned dataset
from pathlib import Path

# Find the ZIP file relative to this Python script
data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "football_data_halfclean.zip"
)

import zipfile

# Open the ZIP and select the actual CSV file
with zipfile.ZipFile(data_path) as z:
    csv_files = [
        name for name in z.namelist()
        if name.lower().endswith(".csv")
        and not name.startswith("__MACOSX/")
        and not name.split("/")[-1].startswith("._")
    ]

    if len(csv_files) != 1:
        raise ValueError(f"Expected one CSV file, found: {csv_files}")

    with z.open(csv_files[0]) as file:
        df = pd.read_csv(
            file,
            encoding="latin1",
            low_memory=False
        )
# 2. Select the variables needed for this analysis
plot_data = df[["HST", "AST", "FTR"]].copy()

# 3. Convert shots on target to numeric values
plot_data["HST"] = pd.to_numeric(
    plot_data["HST"], errors="coerce"
)

plot_data["AST"] = pd.to_numeric(
    plot_data["AST"], errors="coerce"
)

# 4. Remove rows with missing values
plot_data = plot_data.dropna(
    subset=["HST", "AST", "FTR"]
)

# Keep only valid match results
plot_data = plot_data[
    plot_data["FTR"].isin(["H", "D", "A"])
].copy()

# 5. Calculate shots-on-target difference
plot_data["ShotTargetDiff"] = (
    plot_data["HST"] - plot_data["AST"]
)

# 6. Create a boxplot
fig, ax = plt.subplots(figsize=(9, 6))

plot_data.boxplot(
    column="ShotTargetDiff",
    by="FTR",
    ax=ax,
    grid=False
)

ax.set_xlabel("Full-Time Result (A = Away, D = Draw, H = Home)")
ax.set_ylabel("Home Shots on Target - Away Shots on Target")
ax.set_title("Shots on Target Difference by Match Result")

plt.suptitle("")
plt.tight_layout()

# 7. Save the figure
plt.savefig("shots_on_target_boxplot.png", dpi=300)

plt.show()

# 8. Print summary statistics
print(
    plot_data.groupby("FTR")["ShotTargetDiff"].describe()
)

# ==========================================
# Interpretation of EDA Figure 1
# ==========================================

# The boxplot shows a positive association between shots-on-target
# difference (HST - AST) and full-time match results.
#
# Home wins (H) have the highest mean shots-on-target difference
# (approximately 2.55), followed by draws (D) at 0.74.
# Away wins (A) have a negative mean difference (approximately -1.06).
#
# The median shots-on-target difference also increases from
# away wins to draws and then to home wins.
#
# This suggests that shots-on-target difference may be a useful
# variable for classifying full-time match outcomes.
#
# However, the distributions overlap, so this variable alone
# cannot perfectly distinguish between match results.

# ==========================================
# EDA Figure 2: Shot Difference vs Match Result
# ==========================================

# 1. Select the required columns
shot_data = df[["HS", "AS", "FTR"]].copy()

# 2. Convert shot statistics to numeric values
shot_data["HS"] = pd.to_numeric(
    shot_data["HS"], errors="coerce"
)

shot_data["AS"] = pd.to_numeric(
    shot_data["AS"], errors="coerce"
)

# 3. Remove missing values
shot_data = shot_data.dropna(
    subset=["HS", "AS", "FTR"]
)

# Keep only valid match results
shot_data = shot_data[
    shot_data["FTR"].isin(["H", "D", "A"])
].copy()

# 4. Calculate total shot difference
shot_data["ShotDiff"] = (
    shot_data["HS"] - shot_data["AS"]
)

# 5. Create the boxplot
fig, ax = plt.subplots(figsize=(9, 6))

shot_data.boxplot(
    column="ShotDiff",
    by="FTR",
    ax=ax,
    grid=False
)

ax.set_xlabel(
    "Full-Time Result (A = Away, D = Draw, H = Home)"
)

ax.set_ylabel(
    "Home Shots - Away Shots"
)

ax.set_title(
    "Shot Difference by Match Result"
)

plt.suptitle("")
plt.tight_layout()

# 6. Save the figure in BingquanW
output_path = (
    Path(__file__).resolve().parent
    / "shot_difference_boxplot.png"
)

plt.savefig(output_path, dpi=300)
plt.close()

print("\nEDA Figure 2: Shot Difference")
print("Number of matches:", len(shot_data))

# 7. Print summary statistics
print(
    shot_data.groupby("FTR")["ShotDiff"].describe()
)

# ==========================================
# EDA Figure 3
# Corner Difference vs Match Result
# ==========================================

# Select relevant variables
corner_data = df[["HC", "AC", "FTR"]].copy()

# Convert corner statistics to numeric
corner_data["HC"] = pd.to_numeric(
    corner_data["HC"], errors="coerce"
)

corner_data["AC"] = pd.to_numeric(
    corner_data["AC"], errors="coerce"
)

# Remove missing values
corner_data = corner_data.dropna()

# Keep valid match results
corner_data = corner_data[
    corner_data["FTR"].isin(["H", "D", "A"])
].copy()

# Calculate corner difference
corner_data["CornerDiff"] = (
    corner_data["HC"] - corner_data["AC"]
)

# Create boxplot
fig, ax = plt.subplots(figsize=(9, 6))

corner_data.boxplot(
    column="CornerDiff",
    by="FTR",
    ax=ax,
    grid=False
)

ax.set_xlabel("Full-Time Result (A = Away, D = Draw, H = Home)")
ax.set_ylabel("Home Corners - Away Corners")
ax.set_title("Corner Difference by Match Result")

plt.suptitle("")
plt.tight_layout()

# Save figure
plt.savefig(
    "BingquanW/corner_difference_boxplot.png",
    dpi=300
)

plt.show()

# Print summary statistics
print(
    corner_data.groupby("FTR")["CornerDiff"].describe()
)
