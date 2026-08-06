# Function Transformer in Scikit-learn

## Overview
`FunctionTransformer` is used when Scikit-learn's built-in transformers (like `StandardScaler` or `MinMaxScaler`) do not meet your preprocessing needs. It allows you to apply custom mathematical functions to features inside a machine learning pipeline.

In this project, different transformations are applied to the Titanic dataset to reduce skewness and observe their effect on model performance.

---

## Dataset
- **Dataset:** Titanic
- **Features Used:**
  - Age
  - Fare
- **Target:**
  - Survived

---

## Steps Performed
- Loaded the dataset
- Handled missing values using `SimpleImputer`
- Visualized feature distributions using:
  - Distribution Plot
  - QQ Plot
- Trained:
  - Logistic Regression
  - Decision Tree Classifier
- Applied `FunctionTransformer` (`np.log1p`)
- Evaluated models before and after transformation
- Performed 10-Fold Cross Validation
- Applied transformation only on the `Fare` column using `ColumnTransformer`
- Tested custom transformations such as Square Transform

---

## Transformations Used
- Log Transform (`np.log1p`)
- Square Transform (`X ** 2`)

Other commonly used transformations:
- Square Root Transform
- Reciprocal Transform

---

## Models Used
- Logistic Regression
- Decision Tree Classifier

---

## Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn

---

## Key Observation
- Feature transformation improved the performance of **Logistic Regression** by making skewed data closer to a normal distribution.
- Decision Trees are generally less affected by feature transformations because they are not distance-based models.

---

## Learning Outcome
- Learned how to use `FunctionTransformer` for custom preprocessing.
- Learned how to transform selected columns using `ColumnTransformer`.
- Compared model performance before and after feature transformation.
- Understood when mathematical transformations can improve machine learning models.
