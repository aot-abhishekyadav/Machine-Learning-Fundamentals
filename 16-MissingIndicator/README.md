# Missing Indicator

This project demonstrates how to handle missing values using **Missing Indicators** with the Titanic dataset.

### What is Missing Indicator?

A Missing Indicator creates an additional binary column that tells whether the original value was missing (`True`) or not (`False`).

### Workflow

* Load the Titanic dataset.
* Select `Age` and `Fare` as features.
* Split the data into training and testing sets.
* Handle missing values using `SimpleImputer`.
* Create a missing indicator using `MissingIndicator`.
* Add the indicator column (`Age_NA`) to the dataset.
* Train a Logistic Regression model.
* Compare model accuracy.

### Result

* Without missing indicator: **61.45%**
* With missing indicator: **63.13%**

This shows that adding information about missing values can improve model performance.
