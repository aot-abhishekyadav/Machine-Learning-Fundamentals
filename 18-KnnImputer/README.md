# KNN Imputer for Missing Values

## 📌 Overview

This project demonstrates **KNN Imputation** for handling missing values in the Titanic dataset. The `Age` column contains missing values, which are imputed using the K-Nearest Neighbors algorithm.

A **Logistic Regression** model is then used to predict passenger survival.

## 🛠️ Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* KNN Imputer
* Logistic Regression
* GridSearchCV

## 🔄 Workflow

```text
Load Dataset → Check Missing Values → Train-Test Split
→ KNN Imputation → Logistic Regression
→ Hyperparameter Tuning → Evaluation
```

## 📊 Results

| Method               |   Accuracy |
| -------------------- | ---------: |
| Mean Imputation      |     69.27% |
| KNN Imputation       |     70.39% |
| Tuned KNN Imputation | **70.95%** |

### Best Parameters

* `n_neighbors = 2`
* `weights = distance`

## 💡 Key Learning

KNN Imputation performed slightly better than simple mean imputation, and **GridSearchCV** helped find the best KNN parameters.

## 👨‍💻 Author

**Abhishek Yadav**
