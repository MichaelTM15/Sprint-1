# Exploratory Data Analysis
# This section explores relationships between full-time match statistics
# and match outcomes in English football.

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ==============================================
# Exploratory Data Analysis
# Shots on Target Difference vs Match Result
# ==============================================

import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the half-cleaned dataset
df = pd.read_csv(
    "../data/football_data_halfclean.csv",
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
