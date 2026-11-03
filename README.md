# How well can we classify outcomes from Premier League matches using models trained on the lower divisions?

## Members in the group 
- David Alufi
- Michael McDonagh
- Astijus Armonavicius
- Binquan Wang

This project has a 28/24/24/24 equity split in Michael's favour.

# Reading order
All report content is in the directory:
- report/
  
The report takes the following structure:
- 1a.R_Data_Import_and_EDA.Rmd
- 1b.EDA Binquan Wang Report .ipynb 
- 2.R_Missingness_and_SVM_Analysis.Rmd 
- 3.RandomForest.ipynb
- 4.Multinomial_Logistic_Regression.Rmd
- 5.XGBoost_Analysis.pdf
- 6.Wrapup.pdf
  
with html output of the Rmd and ipynb content included for convenience.

# Requirements
Required packages for the files are given at the start of each, they may need to installed before being loaded. For the Rmd files, these consist of:
- gplots, e1071, caret, mice, pROC, ggplot2, patchwork

For the Python files, these consist of:
- pandas, numpy, matplotlib, scikit-learn, IPython, pathlib 

For the SVM file, some RDS files are read from the MichaelM folder, so the working directory should be set to "~GitHub/Sprint-1" or equivalent. Other files may also read from the data and processed-data folders.


# Evidence
Our working is shown in our own directories,
- AstijusA/
- BinquanW/
- DavidA/
- MichaelM/

and the raw and processed data Excel files are found in:
- data/
- processed-data/
  
Michael did the initial EDA and fitted the SVM model, David fitted the XGBoost Model, Astijus fitted the Multinomial Logistic Regression Model, and Binquan fitted the Random Forests and did some extra EDA. Together we wrote the wrapup after examining the entirety of our findings.

# References
- https://football-data.co.uk/data.php
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3900052/
