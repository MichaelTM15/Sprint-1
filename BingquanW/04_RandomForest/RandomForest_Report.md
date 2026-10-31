# Random Forest Model Report

## 1. Introduction

The aim of this project is to investigate whether English football match outcomes can be classified using full-time match statistics and whether adding betting odds improves classification performance.

The target variable is **FTR (Full-Time Result)**, with three possible outcomes:

- **H:** Home win
- **D:** Draw
- **A:** Away win

Random Forest was selected because it combines multiple decision trees and can capture complex, non-linear relationships between football match statistics.

Two models were developed:

1. **Random Forest without Odds:** Using 12 full-time match statistics.
2. **Random Forest with Odds:** Using the same 12 statistics plus three Bet365 betting odds features.

This report describes the dataset, exploratory data analysis, missing-data treatment, model development, evaluation results, feature importance, limitations and conclusions.

## 2. Dataset

The dataset contains historical English football match data obtained from Football-Data.co.uk.

It covers four English football divisions:

- Premier League
- Championship
- League One
- League Two

The processed training and testing datasets were provided by a group member and stored in the GitHub repository:

- `processed-data/football_train_data.csv`
- `processed-data/football_test_data.csv`

| Dataset | Number of matches |
|---|---:|
| Training dataset | 37,824 |
| Testing dataset | 8,740 |
| Total | 46,564 |

The existing training and testing datasets were used without creating an additional random split.

### 2.1 Selected Features

The baseline Random Forest model used 12 numerical predictors representing full-time match statistics.

| Feature | Description |
|---|---|
| HS | Home team shots |
| AS | Away team shots |
| HST | Home team shots on target |
| AST | Away team shots on target |
| HF | Home team fouls |
| AF | Away team fouls |
| HC | Home team corners |
| AC | Away team corners |
| HY | Home team yellow cards |
| AY | Away team yellow cards |
| HR | Home team red cards |
| AR | Away team red cards |

Full-time home and away goals (FTHG and FTAG) were excluded because these variables directly determine the match result. Including them would cause direct target leakage.

### 2.2 Additional Betting Odds Features

A second Random Forest model was developed using the original 12 predictors together with three Bet365 betting odds variables.

| Feature | Description |
|---|---|
| B365H | Bet365 odds for a home win |
| B365D | Bet365 odds for a draw |
| B365A | Bet365 odds for an away win |

The purpose of including these variables was to investigate whether information from betting markets could improve classification accuracy and probability predictions.

The resulting model contained **15 predictors**.

Both models used the same existing test dataset, allowing their performance to be compared.

## 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the relationships between football match statistics and match outcomes.

The analysis examined:

- Shots on target
- Shot differences
- Corner differences
- Home-versus-away attacking performance

The visualisations helped identify relationships between attacking statistics and full-time match results.

These findings supported the selection of numerical predictors for the Random Forest models.

## 4. Data Preprocessing and Missing-Data Analysis

### 4.1 Identifying Missing Values

Before training the baseline Random Forest, the 12 numerical predictors were checked for missing values.

The training dataset contained 16 missing feature values across three observations.

Home and away fouls each contained three missing values, while the other ten selected predictors each contained one missing value.

No missing values were identified in these 12 predictors in the test dataset.

Because only a small number of training observations were affected, several missing-data handling approaches were examined.

### 4.2 Missing-Data Handling Methods

Five approaches were considered.

**Method 1: Complete-case Analysis**

Complete-case analysis removes observations containing missing values in the required predictors.

For the baseline model, three training observations were removed, reducing the training dataset from 37,824 to 37,821 matches.

This method is simple and avoids estimating missing values, although it can introduce bias if missingness is systematic.

**Method 2: Available-case Analysis**

Available-case analysis uses the available observations for each calculation.

In this project, it was applied to descriptive statistics and correlation analysis. It was not treated as a separate Random Forest training method.

**Method 3: Simple Imputation (Mean)**

Mean imputation replaces missing numerical values with the mean of the corresponding feature calculated from the training dataset.

It preserves the number of observations but may underestimate variability and does not account for relationships between features.

**Method 4: Regression Imputation**

Regression imputation estimates missing values using relationships between the incomplete variable and other observed predictors.

It can retain information from related features, although its estimates depend on the regression assumptions.

**Method 5: Multiple Imputation**

Multiple imputation generates several plausible completed datasets to account for uncertainty in missing values.

In this project, five imputed datasets were generated and Random Forest predictions were combined.

This approach is more computationally demanding but provides a way to reflect missing-data uncertainty.

### 4.3 Comparison of Missing-Data Methods

Four approaches were compared through Random Forest classification, while available-case analysis was used for descriptive analysis.

| Method | Accuracy | Macro F1 | Draw Recall |
|---|---:|---:|---:|
| Complete-case Analysis | **57.21%** | **45.76%** | 6.21% |
| Mean Imputation | 56.97% | 45.57% | 6.25% |
| Regression Imputation | 56.99% | 45.54% | 6.07% |
| Multiple Imputation | 57.16% | 45.61% | 5.93% |

Complete-case analysis achieved the highest overall accuracy and Macro F1-score.

However, the performance differences were small. The difference between the highest and lowest accuracy was only 0.24 percentage points.

Because only three training observations were affected, complete-case analysis was selected for the baseline Random Forest model.

This result does not suggest that complete-case analysis is universally superior to imputation. It was simply an appropriate choice for this particular dataset and model configuration.

### 4.4 Missing Values in the Model with Betting Odds

The model with betting odds required all 15 selected predictors to be available.

Since the three additional betting odds variables also contained missing values, complete-case analysis removed 12 training observations in total.

Consequently:

| Dataset | Baseline RF | RF with Odds |
|---|---:|---:|
| Original training observations | 37,824 | 37,824 |
| Removed training observations | 3 | 12 |
| Remaining training observations | 37,821 | 37,812 |
| Test observations | 8,740 | 8,740 |

The number of removed observations remained very small relative to the full training dataset.

The odds-enhanced model therefore used complete-case analysis to maintain a consistent set of available predictors during training.

## 5. Random Forest Model Development

### 5.1 Model Configuration

Both models were implemented using Python and scikit-learn.

The following Random Forest hyperparameters were used:

| Parameter | Value |
|---|---|
| Model | RandomForestClassifier |
| Number of trees | 200 |
| Maximum tree depth | 12 |
| Random state | 42 |
| Parallel processing | -1 |

The random state was fixed to improve reproducibility.

The same model configuration was used in both experiments so that the effect of including betting odds could be examined without changing the main model settings.

### 5.2 Baseline Random Forest

The baseline model was trained using 12 full-time match statistics and 37,821 complete training observations.

It classified match outcomes into home wins, draws and away wins.

The model was evaluated on the existing 8,740 test observations.

### 5.3 Random Forest with Betting Odds

The second model used the same 12 full-time match statistics, with three additional Bet365 betting odds variables.

This increased the number of predictors from 12 to 15.

After complete-case analysis, 37,812 training observations remained.

The purpose of this experiment was to determine whether betting odds could improve the overall classification performance and the quality of predicted class probabilities.

### 5.4 Evaluation Metrics

Several evaluation metrics were used.

**Accuracy** measures the proportion of correctly classified matches.

**Macro F1-score** gives equal importance to the three match-result classes, making it useful when class frequencies differ.

**Draw Recall** measures the proportion of actual drawn matches correctly identified.

**Multiclass ROC-AUC** evaluates how effectively the model distinguishes between the three match outcomes using predicted probabilities. The One-vs-Rest (OvR) approach with macro averaging was used.

**Multiclass Log Loss** measures the quality of the predicted class probabilities. Lower values indicate better probabilistic predictions.

**Akaike Information Criterion (AIC)** was considered for the group comparison. However, standard AIC is not directly applicable to Random Forest because the model does not have a conventional likelihood-based formulation and a clearly defined number of estimated parameters. AIC was therefore reported as N/A.

## 6. Model Evaluation and Comparison

### 6.1 Overall Model Performance

The two Random Forest models produced the following results:

| Metric | Random Forest | Random Forest with Odds |
|---|---:|---:|
| Accuracy | 57.21% | **59.61%** |
| Macro F1-score | 45.76% | **47.41%** |
| Draw Recall | **6.21%** | 5.14% |
| Multiclass ROC-AUC (OvR, Macro) | 0.7134 | **0.7419** |
| Multiclass Log Loss | 0.9240 | **0.8839** |
| AIC | N/A | N/A |

The Random Forest model with betting odds achieved better overall performance across Accuracy, Macro F1, ROC-AUC and Log Loss.

### 6.2 Accuracy Comparison

The baseline Random Forest achieved an accuracy of 57.21%.

After including betting odds, accuracy increased to 59.61%.

This represents an improvement of **2.40 percentage points**.

The result suggests that betting odds provide additional information that is useful for classifying match outcomes.

### 6.3 Multiclass ROC-AUC Comparison

Multiclass ROC-AUC was calculated using the One-vs-Rest approach and macro averaging.

The baseline model achieved:

**ROC-AUC = 0.7134**

The model with betting odds achieved:

**ROC-AUC = 0.7419**

The improvement of 0.0285 indicates that the odds-enhanced model had better overall discrimination between the three match-result classes.

This supports the accuracy comparison and suggests that betting odds improve the model's ability to distinguish among home wins, draws and away wins.

### 6.4 Multiclass Log Loss Comparison

Multiclass Log Loss was calculated using the predicted probabilities for all three match outcomes.

The baseline model achieved:

**Log Loss = 0.9240**

The model with betting odds achieved:

**Log Loss = 0.8839**

Since lower Log Loss is better, the odds-enhanced model produced better probabilistic predictions on the test dataset.

This metric is also useful for the group's comparison between Random Forest and XGBoost, provided that the models use consistent outcome labels and test data.

### 6.5 Draw Classification

Despite improving overall model performance, adding betting odds did not improve the detection of draws.

Draw recall decreased from 6.21% to 5.14%.

This means that both models correctly identified only a small proportion of actual drawn matches.

The result highlights an important limitation: better overall accuracy and probability metrics do not necessarily mean improved classification performance for every outcome.

### 6.6 Confusion Matrix

Confusion matrices were used to compare actual and predicted match outcomes.

The baseline model performed better at identifying home wins than draws.

Many drawn matches were incorrectly classified as home wins or away wins.

The model with betting odds also struggled to distinguish drawn matches reliably.

The confusion matrices are saved as:

- `rf_complete_case_confusion_matrix.png`
- `rf_with_odds_confusion_matrix.png`

## 7. Feature Importance

### 7.1 Baseline Random Forest

Random Forest feature importance was examined to understand which match statistics contributed most strongly to the model's decisions.

The five most important predictors in the baseline model were approximately:

| Feature | Importance |
|---|---:|
| HST — Home shots on target | 22.45% |
| AST — Away shots on target | 19.97% |
| HC — Home corners | 8.10% |
| HS — Home shots | 8.03% |
| AS — Away shots | 7.68% |

Home and away shots on target were the two most influential predictors.

This suggests that attacking performance played an important role in the model's classifications.

### 7.2 Random Forest with Betting Odds

After including betting odds, home and away shots on target remained influential predictors.

The betting odds variables also contributed to the model.

In particular, B365H, B365D and B365A added information about the relative likelihood of different match outcomes as assessed by the betting market.

The inclusion of these variables was associated with higher accuracy and AUC and lower Log Loss.

However, feature importance should not be interpreted as proof of a causal relationship.

The feature importance visualisations are saved as:

- `rf_complete_case_feature_importance.png`
- `rf_with_odds_feature_importance.png`

## 8. Limitations

### 8.1 Difficulty Predicting Draws

The main limitation of both Random Forest models was their poor ability to identify draws.

The baseline model achieved a draw recall of 6.21%, while the odds-enhanced model achieved 5.14%.

This suggests that predicting draws remains challenging even when betting odds are included.

Future work could investigate class weighting, alternative features and other approaches to handling class imbalance.

### 8.2 Missing-Data Assumptions

Complete-case analysis removes observations with missing predictors.

Although only three observations were removed from the baseline training dataset and 12 from the odds-enhanced training dataset, this approach may introduce bias when missingness is not random.

The missing-data comparison showed that the four training approaches produced similar results for the baseline model.

However, the same comparison was not conducted for every missing-data method in the odds-enhanced model.

### 8.3 Full-Time Statistics and Predictive Interpretation

Both Random Forest models used full-time match statistics, including shots, fouls, corners and cards.

These variables are only fully available after a match has finished.

Therefore, the models perform **retrospective match-outcome classification rather than genuine pre-match forecasting**.

Although betting odds may be available before kick-off, their inclusion does not make the overall odds-enhanced model a pre-match forecasting system because it still uses full-time match statistics.

A genuine pre-match prediction model would need to rely exclusively on variables available before the match.

### 8.4 Hyperparameter Optimisation

Both models used fixed hyperparameters rather than an extensively optimised configuration.

Their performance may therefore not represent the maximum achievable performance of Random Forest.

Future work could use stratified k-fold cross-validation and hyperparameter tuning to examine different tree depths, minimum leaf sizes and class weights.

The test dataset should remain reserved for final model evaluation.

### 8.5 Model Comparison and AIC

Standard AIC was not calculated for the Random Forest models because they do not provide the usual likelihood-based parameterisation needed for a conventional AIC calculation.

Instead, Accuracy, Macro F1, Multiclass ROC-AUC and Multiclass Log Loss were used to evaluate performance.

Log Loss provides a particularly useful metric for comparing probability-based predictions from Random Forest and XGBoost.

However, cross-model comparisons should account for differences in feature availability, model inputs and evaluation procedures.

## 9. Conclusion

This project developed and evaluated two Random Forest models for classifying English football match outcomes into home wins, draws and away wins.

The baseline model used 12 full-time match statistics, while the second model included three additional Bet365 betting odds variables.

Five missing-data handling approaches were considered. Four were evaluated using Random Forest classification, while available-case analysis was used for descriptive statistics and correlation analysis. Complete-case analysis was selected for the baseline model because it achieved slightly higher accuracy and Macro F1-score while removing only three training observations.

The baseline Random Forest achieved an accuracy of **57.21%**, a multiclass ROC-AUC of **0.7134** and a Log Loss of **0.9240**.

After including betting odds, accuracy increased to **59.61%**, ROC-AUC improved to **0.7419**, and Log Loss decreased to **0.8839**. Macro F1-score also increased from 45.76% to 47.41%.

These results suggest that betting odds provide useful additional information and improve the overall classification performance and probability estimates of the Random Forest model.

However, both models performed poorly when identifying draws. Draw recall remained below 7%, indicating that this outcome was frequently confused with home or away wins.

Another important limitation is that both models use full-time match statistics. Therefore, they should be interpreted as retrospective classification models rather than genuine pre-match forecasting systems.

Overall, **the Random Forest model with betting odds performed better across the main overall evaluation metrics**, although predicting draws remains an important weakness. Future improvements could involve class balancing, hyperparameter tuning through cross-validation and developing a separate model using exclusively pre-match features.

The results provide a basis for comparison with other machine learning approaches, particularly XGBoost, using consistent evaluation metrics such as Accuracy, ROC-AUC and Multiclass Log Loss.
