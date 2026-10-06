# Premier League Match Outcome Analysis

## 1. Data Preparation

### Research Question

**Can we predict the outcome of a Premier League match from the full-time statistics?**

This analysis uses historical Premier League match data from multiple seasons.
Each row represents one Premier League match. The aim is to investigate whether
full-time match statistics, such as shots, shots on target, corners, fouls and
cards, can be used to classify the full-time match result.
import pandas as pd
import matplotlib.pyplot as plt
season_22_23 = pd.read_csv("../data/2022_23.csv")
season_23_24 = pd.read_csv("../data/2023_24.csv")
season_24_25 = pd.read_csv("../data/2024_25.csv")
season_25_26 = pd.read_csv("../data/2025_26.csv")
season_26_27 = pd.read_csv("../data/2026_27.csv")
