# EDA and Modelling Plan

## Aim

My contribution will focus on exploratory data analysis (EDA) of the football
match statistics, followed by a decision tree classification model.

The EDA will investigate relationships between full-time match statistics and
the full-time result (`FTR`), where:

- `H` = Home win
- `D` = Draw
- `A` = Away win

The purpose of the EDA is to identify potentially useful relationships before
the data is cleaned and used for modelling.

## Planned Exploratory Data Analysis

## Dataset and Current Progress

The dataset was provided by Michael as a partially cleaned
version of the combined football match data.

It contains historical match statistics from four English
football leagues:

- Premier League (E0)
- Championship (E1)
- League One (E2)
- League Two (E3)

The original CSV file was too large to upload directly to GitHub,
so a compressed ZIP version has been uploaded to the `data` folder.

Dataset location:
`data/football_data_halfclean.zip`

The dataset is suitable for preliminary exploratory data analysis
(EDA), but further cleaning will be required before training
the classification model.

My next step is to produce visualisations examining the
relationships between match statistics and full-time results.

### 1. Shots on Target Difference and Match Result

Calculate the difference between home and away shots on target:

`HST - AST`

A boxplot will be used to compare this difference across home wins, draws and
away wins.

**Question:** Is the difference in shots on target associated with the final
match result?

### 2. Shot Difference and Match Result

Calculate the difference between total home and away shots:

`HS - AS`

A boxplot will be used to examine how shot difference varies between the three
match outcomes.

**Question:** Is total shot difference associated with the final match result?

### 3. Corner Difference and Match Result

Calculate:

`HC - AC`

and compare the distribution across home wins, draws and away wins.

**Question:** Is the difference in corners associated with match outcome?

### 4. Home vs Away Shots on Target

A scatter plot will compare home shots on target (`HST`) with away shots on
target (`AST`), with matches grouped by their full-time result.

**Question:** Do home wins, draws and away wins show different patterns of
shots on target?

## Planned Modelling

After EDA and data cleaning, I plan to use a decision tree classifier to
classify the full-time result (`FTR`) using selected match statistics.

The model will use lower-division English football data for training and
Premier League data for testing, allowing us to investigate whether patterns
learned from lower divisions generalise to the Premier League.

Final-score variables such as `FTHG` and `FTAG` will not be used as predictors
because they directly determine `FTR` and would introduce data leakage.

Model performance will be evaluated using measures such as accuracy and a
confusion matrix. The structure of the decision tree and feature importance
may also be examined to help interpret the model.

## Resources

### Football-Data.co.uk

Football-Data.co.uk is used as the main data source for this project. It provides
historical English football match results, match statistics and betting odds
across multiple divisions and seasons.

The data from the Premier League, Championship, League One and League Two will
be used for the exploratory analysis and subsequent modelling.
