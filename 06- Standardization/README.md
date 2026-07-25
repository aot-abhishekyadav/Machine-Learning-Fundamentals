# 📊 Feature Scaling using Standardization in Machine Learning

This project demonstrates the concept of **Feature Scaling using Standardization** with Python and Scikit-learn. The project explains how StandardScaler transforms numerical features so that they have a **mean of 0 and a standard deviation of 1**.

The implementation uses the **Social Network Ads dataset** and compares machine learning model performance before and after feature scaling.

---

## 📌 Project Overview

Machine Learning datasets often contain features with very different scales. For example:

* **Age:** 18–60
* **Estimated Salary:** ₹15,000–₹1,50,000

When features have different ranges, some machine learning algorithms can be influenced by the magnitude of the features.

This project demonstrates how **Standardization** can bring numerical features to a similar scale and explains when feature scaling is important.

---

## 🧠 What is Standardization?

Standardization is a feature scaling technique that transforms data using the following formula:

**Z = (X − μ) / σ**

Where:

* **X** = Original value
* **μ** = Mean of the feature
* **σ** = Standard deviation
* **Z** = Standardized value

After standardization:

* Mean ≈ **0**
* Standard deviation ≈ **1**

---

## 🎯 Objectives

The main objectives of this project are:

* Understand Feature Scaling
* Learn Standardization
* Implement `StandardScaler`
* Understand `fit()`, `transform()` and `fit_transform()`
* Avoid Data Leakage during scaling
* Compare data before and after scaling
* Visualize the effect of scaling
* Compare model performance with and without scaling
* Understand the effect of outliers on Standardization

---

## 📂 Dataset

The project uses the **Social Network Ads dataset**.

### Features

| Feature         | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| Age             | Age of the customer                                          |
| EstimatedSalary | Estimated salary of the customer                             |
| Purchased       | Target variable indicating whether the product was purchased |

The `User ID` and `Gender` columns are removed for this experiment.

---

## 🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 🔄 Project Workflow

```text
Load Dataset
     ↓
Data Cleaning
     ↓
Select Features
     ↓
Train-Test Split
     ↓
Apply StandardScaler
     ↓
Transform Train & Test Data
     ↓
Visualize Before vs After Scaling
     ↓
Train Machine Learning Models
     ↓
Compare Model Performance
     ↓
Analyze Effect of Outliers
```

---

## ⚙️ StandardScaler Implementation

The dataset is first divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1),
    df['Purchased'],
    test_size=0.3,
    random_state=0
)
```

Then StandardScaler is applied:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform both training and testing data. This helps prevent **Data Leakage**.

---

## 🔍 Understanding `fit()`, `transform()` and `fit_transform()`

### `fit()`

Learns the mean and standard deviation from the training data.

```python
scaler.fit(X_train)
```

### `transform()`

Uses the learned parameters to transform the data.

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### `fit_transform()`

Combines both operations.

```python
X_train_scaled = scaler.fit_transform(X_train)
```

### Best Practice

```text
Training Data → fit_transform()
Testing Data  → transform()
```

Never fit the scaler separately on test data.

---

## 📊 Before vs After Standardization

### Before Scaling

The original features have different ranges:

* Age → approximately 18 to 60
* EstimatedSalary → approximately ₹15,000 to ₹1,50,000

### After Scaling

Both features are transformed to a similar scale:

* Mean ≈ 0
* Standard Deviation ≈ 1

This makes the numerical features comparable for algorithms that are sensitive to feature magnitude.

---

## 🤖 Machine Learning Models

The project experiments with:

### 1. Logistic Regression

Logistic Regression is sensitive to feature scale, so standardization can be beneficial.

The model is trained using:

* Original features
* Standardized features

The accuracy is then compared.

### 2. Decision Tree

Decision Trees generally do not require feature scaling because they make decisions based on feature thresholds.

This project demonstrates that scaling usually does not provide a significant advantage for tree-based models.

---

## 📈 Model Performance

The Logistic Regression experiment produced the following accuracy results:

| Model                                 | Accuracy |
| ------------------------------------- | -------: |
| Logistic Regression (Without Scaling) |   87.50% |
| Logistic Regression (With Scaling)    |   86.67% |

> Note: Scaling does not always increase accuracy. Its main purpose is to put features on a comparable scale, especially for scale-sensitive algorithms.

---

## 📌 Algorithms That Usually Benefit from Scaling

Feature scaling is generally useful for:

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machines (SVM)
* Neural Networks
* Principal Component Analysis (PCA)
* K-Means Clustering

---

## 📌 Algorithms That Usually Don't Require Scaling

Scaling is generally not necessary for:

* Decision Trees
* Random Forest
* XGBoost
* LightGBM
* CatBoost

These algorithms are generally based on feature splits or tree structures and are less sensitive to feature magnitude.

---

## 📊 Visualizations

The project includes visualizations for:

* Feature distribution before scaling
* Feature distribution after scaling
* Scatter plot before scaling
* Scatter plot after scaling
* Age distribution comparison
* Salary distribution comparison
* Probability distribution comparison

These visualizations help understand how Standardization changes the scale while preserving the overall distribution shape.

---

## 🚨 Effect of Outliers

The project also demonstrates how outliers can affect Standardization.

Additional extreme values are introduced into:

* `Age`
* `EstimatedSalary`

The effect of these outliers is then visualized using scatter plots.

This demonstrates an important limitation of StandardScaler:

> Standardization is sensitive to outliers because mean and standard deviation are affected by extreme values.

For datasets with significant outliers, **RobustScaler** may be a better alternative.

---

## 💡 Key Learnings

Through this project, I learned:

* What Feature Scaling is
* Why Standardization is required
* How `StandardScaler` works
* Difference between `fit()` and `transform()`
* How to prevent Data Leakage
* Difference between scaled and unscaled data
* Which ML algorithms require scaling
* Why tree-based models generally don't require scaling
* How scaling affects model training
* How outliers affect Standardization
* Importance of preprocessing in Machine Learning

---

## 🚀 Future Improvements

The project can be extended by:

* Comparing `StandardScaler` with `MinMaxScaler`
* Using `RobustScaler` for outlier-heavy data
* Building a complete Scikit-learn `Pipeline`
* Comparing multiple classification algorithms
* Adding cross-validation
* Performing hyperparameter tuning
* Evaluating models using Precision, Recall and F1-score
* Adding a confusion matrix and ROC-AUC curve

---

## 👨‍💻 Author

**Abhishek Yadav**

Aspiring Data Scientist | Machine Learning Enthusiast | Python Developer

---

## ⭐ If you found this project useful

Feel free to ⭐ star this repository and connect with me on LinkedIn!
