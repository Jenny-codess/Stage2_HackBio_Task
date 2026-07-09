# Stage2_HackBio_Task
This is the stage 2 task from HackBio internship in AI for Genomics.

## Predicting Drug Sensitivity in Cancer (GDSC)

Using the same dataset as last week, build a machine learning models to predict the sensitivityof cancer cell lines to treatments using the ln(IC50) column.

There are two possible ways to approach this task. Either approach it as a classification task or regression task. If you are approaching it as a regression task, simply use the column LN_IC50 column.

If you are approaching it as a classification task i.e. to predict resistant and sensitive readings, we will perform a quantile-based split. Basically, anything below 25th percentile is sensitive and anything above 75% is resistant. Everything in between is discarded. You can adjust the thresholds if you like. See code below

lower = df["ln_IC50"].quantile(0.25) #0.25 can be adjust to any number between 0 and 1

upper = df["ln_IC50"].quantile(0.75) #0.75 can be adjust to any number between 0 and 1

Modeling: We suggest you try either random forest or xgboosting. They both have regression and classification methods.

Feature Importance and Evaluation: These are key components of your report and our grading. Justify everything like a biologist, don’t just look for numbers that pass a threshold.
