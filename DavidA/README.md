This repository contains the machine learning pipeline for predicting full-time football match outcomes (Home Win, Draw, Away Win) using an optimized XGBoost Classifier. The model was trained on a comprehensive historical dataset of ~37,800 matches, utilizing match statistics and Bet365 betting odds to establish predictive baselines.
Cross-Validation Accuracy: 55.6% (±0.0109) across 5-Fold Cross-Validation with parallel processing.
Feature Importance: Primary predictive drivers include market odds (Bet365), historical home/away form, and shot metrics (shots on target).
Confusion Matrix & Draw Volatility:
The model demonstrates strong predictive power for Home Wins (2,657 correct) and Away Wins (1,360 correct).
As expected in sports analytics, Draws remain challenging due to high outcome variance and class imbalance (229 correctly identified draws out of 2,061 total actual draws).

