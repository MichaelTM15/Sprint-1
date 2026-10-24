# Random Forest Model Report

## 1. Introduction

The aim of this project is to investigate whether the outcome of an English football match can be classified using full-time match statistics. The target variable is **FTR (Full-Time Result)**, which has three possible outcomes:

- **H:** Home win
- **D:** Draw
- **A:** Away win

A Random Forest classifier was selected because it combines multiple decision trees and can capture complex, non-linear relationships between match statistics.

This report describes the dataset, exploratory analysis, missing-data treatment, model development, evaluation results, feature importance, and limitations.

## 2. Dataset

The dataset contains historical English football match statistics obtained from Football-Data.co.uk.

The overall project includes data from four English football divisions:

- Premier League
- Championship
- League One
- League Two

The processed training and testing datasets were provided by a group member and stored in the GitHub repository.

The following files were used:

- `processed-data/football_train_data.csv`
- `processed-data/football_test_data.csv`

| Dataset | Number of Matches |
|---|---:|
| Training dataset | 37,824 |
| Testing dataset | 8,740 |
| Total | 46,564 |

The existing training and testing datasets were used without creating an additional random split.

### 2.1 Selected Features

The Random Forest model used 12 numerical predictors representing full-time match statistics.

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

Betting odds were excluded from the model because they contained substantial missing data and were not necessary for the selected modelling approach.

Full-time home and away goals (FTHG and FTAG) were also excluded because they directly determine the match result. Including these variables would cause target leakage and make the prediction task trivial.

## 3. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was conducted to understand the relationships between match statistics and football match outcomes.

The analysis examined shots on target, shot differences, corner differences, and home-versus-away attacking performance.

The visualisations were used to explore potential relationships between these statistics and the full-time result.

These findings helped identify relevant numerical predictors for the Random Forest classifier.

## 4. Data Preprocessing and Missing-Data Analysis

### 4.1 Identifying Missing Values

Before training the model, the selected numerical predictors were checked for missing values.

The training dataset contained 16 missing feature values distributed across three observations. Specifically, the home and away foul variables each contained three missing values, while each of the other ten selected features contained one missing value.

No missing values were identified in the 12 selected predictors in the test dataset.

Because the number of affected observations was very small relative to the training dataset, several missing-data handling methods were compared to determine whether the choice of method influenced model performance.

### 4.2 Missing-Data Handling Methods

Four approaches were evaluated using the Random Forest classifier.

**Method 1: Complete-case Analysis**

Complete-case analysis removes observations containing missing values in the required predictor variables.

In this project, three incomplete training observations were removed, reducing the training dataset from 37,824 to 37,821 matches.

This method avoids introducing estimated values but may cause bias if the missingness mechanism is not completely at random.

**Method 2: Simple Imputation (Mean)**

Simple imputation replaces missing numerical values with the mean of the corresponding feature calculated from the training data.

This method retains all training observations and is computationally efficient. However, it does not account for relationships between variables or uncertainty in the missing values.

**Method 3: Regression Imputation**

Regression imputation estimates missing values using information from other observed predictors.

This approach can account for relationships between variables and may produce more informed estimates than simple mean imputation.

However, its effectiveness depends on the assumptions and predictive accuracy of the imputation model.

**Method 4: Multiple Imputation**

Multiple imputation generates several completed datasets by estimating missing values multiple times.

In this experiment, five imputations were performed, and the resulting Random Forest predictions were combined.

This approach reflects uncertainty in the missing values more explicitly than single imputation, but it requires additional computational resources.

All four approaches used the training data for their respective missing-data treatments. The original test dataset was retained for evaluation.

### 4.3 Comparison of Missing-Data Methods

The following results were obtained from the four Random Forest experiments.

| Method | Accuracy | Macro F1 | Draw Recall |
|---|---:|---:|---:|
| **Complete-case Analysis** | **57.21%** | **45.76%** | 6.21% |
| Simple Imputation (Mean) | 56.97% | 45.57% | **6.25%** |
| Regression Imputation | 56.99% | 45.54% | 6.07% |
| Multiple Imputation | 57.16% | 45.61% | 5.93% |

Complete-case analysis achieved the highest overall accuracy and Macro F1-score among the four methods.

Mean imputation achieved a slightly higher recall for drawn matches, although the difference was very small.

Multiple imputation produced results close to those of complete-case analysis, while regression imputation and mean imputation achieved slightly lower overall accuracy.

The difference between the highest and lowest accuracy was only 0.24 percentage points.

These results suggest that the choice of missing-data treatment had a relatively small influence on Random Forest performance in this experiment.

### 4.4 Selection of the Final Method

Complete-case analysis was selected for the final Random Forest model for three reasons.

First, it achieved the highest accuracy and Macro F1-score in the comparison.

Second, only three observations needed to be removed from a training dataset containing 37,824 matches, resulting in a negligible reduction in sample size.

Third, it avoided introducing estimated feature values and was simpler to implement and interpret than regression or multiple imputation.

However, the small performance differences do not establish that complete-case analysis is universally superior. The comparison is specific to this dataset and model configuration.

The final Random Forest model therefore used 37,821 complete training observations and the original 8,740 test observations.

## 5. Random Forest Model Development

### 5.1 Model Configuration

A Random Forest classifier was implemented using Python and scikit-learn.

The model used the following hyperparameters:

| Parameter | Value |
|---|---|
| Model | RandomForestClassifier |
| Number of trees (`n_estimators`) | 200 |
| Maximum tree depth (`max_depth`) | 12 |
| Random state | 42 |
| Parallel processing (`n_jobs`) | -1 |

The model was trained using the 12 selected match statistics to classify the full-time result into home win, draw, or away win.

The random state was fixed to improve reproducibility.

### 5.2 Training and Testing

The model was trained using the complete-case training dataset.

Predictions were then generated for the existing test dataset.

Model performance was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

Macro F1-score was included because the three match-result classes are not equally represented in the dataset.

Draw recall was examined separately because identifying drawn matches was particularly challenging.

## 6. Model Evaluation

### 6.1 Overall Performance

The final Random Forest model achieved the following results:

| Metric | Result |
|---|---:|
| Accuracy | 57.21% |
| Macro F1-score | 45.76% |
| Draw Recall | 6.21% |
| Training observations | 37,821 |
| Test observations | 8,740 |

The model correctly classified approximately 57.21% of the test matches.

For comparison, the original median-imputation baseline achieved approximately 57.16% accuracy. The complete-case model therefore improved accuracy by only 0.05 percentage points.

This small difference indicates that changing the missing-data handling method did not substantially improve predictive performance.

### 6.2 Classification Performance

The model performed better at identifying home wins than draws.

Home wins were the most common outcome in the test dataset, followed by away wins and draws.

Draw recall was approximately 6.21%, indicating that the model correctly identified only a small proportion of actual drawn matches.

This suggests that the Random Forest classifier struggled to distinguish drawn matches from home or away wins using the selected full-time statistics.

Although overall accuracy exceeded 57%, the low draw recall demonstrates that accuracy alone does not provide a complete assessment of model performance.

### 6.3 Confusion Matrix

The confusion matrix was used to compare actual and predicted match outcomes across the three classes.

The results showed that home wins were identified more successfully than draws.

A substantial number of actual drawn matches were classified as home wins or away wins.

This pattern suggests that the model favoured more common or easier-to-identify outcomes rather than consistently recognising all three classes.

The confusion matrix is saved as:

`rf_complete_case_confusion_matrix.png`

## 7. Feature Importance

Random Forest feature importance was used to investigate which match statistics contributed most strongly to the model's decisions.

The most important predictors were:

| Feature | Importance (approximately) |
|---|---:|
| HST — Home shots on target | 22.45% |
| AST — Away shots on target | 19.97% |
| HC — Home corners | 8.10% |
| HS — Home shots | 8.03% |
| AS — Away shots | 7.68% |

Home and away shots on target were the two most influential features.

This suggests that attacking performance, particularly the number of shots on target, was strongly associated with how the Random Forest classified match outcomes.

Other statistics, including corners, fouls, and disciplinary records, also contributed to the model.

However, Random Forest feature importance does not establish causal relationships between these statistics and match results.

The feature importance visualisation is saved as:

`rf_complete_case_feature_importance.png`

## 8. Limitations

### 8.1 Difficulty Predicting Draws

The main limitation of the Random Forest classifier was its poor performance in identifying drawn matches.

Despite achieving approximately 57% overall accuracy, draw recall remained close to 6%.

This indicates that the model was much less effective for one outcome category than for the others.

Future improvements could investigate class weighting, alternative feature engineering, and hyperparameter optimisation.

### 8.2 Missing-Data Assumptions

Complete-case analysis assumes that removing incomplete observations does not introduce substantial bias.

If the missingness mechanism is related to observed or unobserved characteristics, deleting incomplete observations may affect the representativeness of the training data.

However, only three training observations were removed in this project, so the loss of sample size was minimal.

The comparison of four missing-data methods also showed only small differences in predictive performance.

### 8.3 Full-Time Statistics and Target Leakage

The model uses full-time match statistics, including shots, fouls, corners, and cards.

These variables are only fully available after a match has finished.

Therefore, this model performs retrospective match-outcome classification rather than genuine pre-match forecasting.

Although final goal counts were excluded to avoid direct target leakage, the model should not be interpreted as a system that can predict match outcomes before kick-off.

A genuine pre-match forecasting model would require features available before the match, such as historical team performance, previous results, and pre-match team information.

### 8.4 Model Configuration

The Random Forest model used a fixed set of hyperparameters rather than an extensively optimised configuration.

As a result, its performance may not represent the best achievable performance of the Random Forest algorithm.

Future work could use stratified k-fold cross-validation and hyperparameter tuning to evaluate different tree depths, minimum leaf sizes, and class-weighting strategies.

Such tuning should be performed using training data, with the test dataset reserved for final evaluation.

## 9. Conclusion

This project developed a Random Forest classifier to classify English football match outcomes using 12 full-time match statistics.

Four missing-data handling approaches were compared: complete-case analysis, mean imputation, regression imputation, and multiple imputation.

Complete-case analysis achieved the highest overall accuracy of 57.21% and Macro F1-score of 45.76%. It was selected as the final method because only three training observations contained missing predictors and the method avoided introducing estimated values.

The comparison demonstrated that missing-data treatment had only a minor effect on model performance for this dataset.

Shots on target were the most important predictors, suggesting that attacking statistics played a major role in the model's classifications.

However, the model struggled to identify draws, with recall of approximately 6.21%. In addition, because the predictors were recorded during the match, the model cannot be used for genuine pre-match forecasting.

Future work should focus on improving draw classification, investigating alternative features, and optimising model hyperparameters using cross-validation.

Overall, the Random Forest model provided a useful baseline for investigating the relationship between full-time football statistics and match outcomes.
