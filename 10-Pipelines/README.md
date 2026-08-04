# Machine Learning Prediction: With Pipeline vs Without Pipeline

This repository demonstrates how to build a machine learning prediction workflow using two different approaches:

1. Prediction without using a Pipeline
2. Prediction using a Scikit-learn Pipeline

The main purpose of this project is to understand how preprocessing, feature transformation, model training, and prediction work in both approaches.

---

## Project Objective

The objective of this project is to compare the traditional manual machine learning workflow with the Pipeline-based workflow.

The project uses the Titanic dataset to perform classification and predict whether a passenger survived or not.

---

## Dataset

The Titanic dataset contains passenger information such as:

* Pclass
* Sex
* Age
* SibSp
* Parch
* Fare
* Embarked

Target column:

* Survived

---

## Concepts Used

* SimpleImputer
* OneHotEncoder
* MinMaxScaler
* ColumnTransformer
* Pipeline
* DecisionTreeClassifier
* GridSearchCV
* Pickle
* Train-test split
* Model prediction

---

## Repository Structure

```text
ML-Pipeline-Learning/
│
├── 01_Without_Pipeline.ipynb
├── 02_With_Pipeline.ipynb
├── pipe.pkl
├── README.md
└── models/
    ├── ohe_sex.pkl
    ├── ohe_embarked.pkl
    └── clf.pkl
```

---

# Prediction Without Pipeline

In the without-pipeline approach, every preprocessing step is performed manually.

The typical flow is:

```text
Raw Data
   ↓
Missing Value Imputation
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Combining Transformed Columns
   ↓
Model Training
   ↓
Prediction
```

Example:

```python
X_train_age = imputer.fit_transform(X_train[['Age']])
X_test_age = imputer.transform(X_test[['Age']])

X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_test_sex = ohe_sex.transform(X_test[['Sex']])

X_train_embarked = ohe_embarked.fit_transform(X_train[['Embarked']])
X_test_embarked = ohe_embarked.transform(X_test[['Embarked']])
```

After preprocessing, all transformed columns are manually combined:

```python
X_train_transformed = np.concatenate(
    (
        X_train_age,
        X_train_sex,
        X_train_embarked,
        X_train_remaining
    ),
    axis=1
)
```

Then the model is trained:

```python
clf.fit(X_train_transformed, y_train)
```

For prediction, the new input must also pass through all the same preprocessing steps manually.

---

## Problems With the Manual Approach

The without-pipeline approach works, but it has some disadvantages:

* More code is required.
* Preprocessing steps must be applied manually.
* There is a higher chance of making mistakes.
* Train and test transformations may become inconsistent.
* Multiple objects need to be saved separately.
* Deployment becomes more difficult.
* Data leakage can happen if preprocessing is performed incorrectly.

For example, separate objects may need to be saved:

```python
pickle.dump(ohe_sex, open('ohe_sex.pkl', 'wb'))
pickle.dump(ohe_embarked, open('ohe_embarked.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))
```

---

# Prediction Using Pipeline

In the Pipeline approach, preprocessing and the machine learning model are connected in one sequence.

The typical flow is:

```text
Raw Data
   ↓
ColumnTransformer
   ↓
Missing Value Imputation
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Prediction
```

Example:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('imputer', trf1),
    ('encoder', trf2),
    ('scaler', trf3),
    ('model', clf)
])
```

The complete pipeline can be trained using:

```python
pipe.fit(X_train, y_train)
```

The Pipeline automatically performs:

```text
fit_transform() on preprocessing steps
fit() on the final model
```

For prediction:

```python
y_pred = pipe.predict(X_test)
```

The Pipeline automatically performs:

```text
transform() on preprocessing steps
predict() using the trained model
```

---

# Prediction on New Input

With Pipeline, prediction on a new passenger becomes simple.

```python
new_passenger = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 'male',
    'Age': 25,
    'SibSp': 0,
    'Parch': 0,
    'Fare': 7.25,
    'Embarked': 'S'
}])

prediction = pipe.predict(new_passenger)

print(prediction)
```

There is no need to manually apply imputation, encoding, or scaling because the Pipeline handles everything automatically.

---

# Difference Between Prediction With and Without Pipeline

| Without Pipeline                                        | With Pipeline                                            |
| ------------------------------------------------------- | -------------------------------------------------------- |
| Preprocessing is performed manually                     | Preprocessing is performed automatically                 |
| More code is required                                   | Less and cleaner code                                    |
| Higher chance of mistakes                               | Lower chance of mistakes                                 |
| Train and test transformations must be managed manually | Train and test transformations are handled automatically |
| Multiple objects may need to be saved                   | The complete workflow can be saved in one file           |
| Deployment is more difficult                            | Deployment is easier                                     |
| Data leakage risk is higher                             | Pipeline helps prevent data leakage                      |
| GridSearchCV integration is more complicated            | GridSearchCV works directly with Pipeline                |
| New input must be transformed manually                  | Raw input can be passed directly to the Pipeline         |

---

# GridSearchCV With Pipeline

Pipeline can be directly used inside GridSearchCV.

```python
from sklearn.model_selection import GridSearchCV

params = {
    'decisiontreeclassifier__max_depth': [1, 2, 3, 4, 5, None]
}

grid = GridSearchCV(
    pipe,
    params,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)
```

GridSearchCV performs preprocessing separately inside every cross-validation fold.

This helps reduce data leakage.

---

# Saving the Model

Without Pipeline, multiple preprocessing objects and the model may need to be saved separately.

With Pipeline, the complete workflow can be saved in one file.

```python
import pickle

with open('pipe.pkl', 'wb') as file:
    pickle.dump(pipe, file)
```

Load the saved Pipeline:

```python
with open('pipe.pkl', 'rb') as file:
    pipe = pickle.load(file)
```

Prediction after loading:

```python
prediction = pipe.predict(new_passenger)
```

---

# Main Learning

The without-pipeline approach is useful for understanding how every preprocessing step works internally.

The Pipeline approach is better for real-world machine learning projects because it provides:

* Clean code
* Consistent preprocessing
* Easy model training
* Easy prediction
* Easy hyperparameter tuning
* Easy model deployment
* Lower risk of data leakage

---

# Conclusion

Both approaches produce predictions, but the Pipeline approach is more organized, reusable, and production-friendly.

The manual approach helped me understand each preprocessing step separately, while the Pipeline approach showed me how Scikit-learn can combine preprocessing and model training into one complete workflow.

This repository was created for learning and understanding the practical difference between prediction using Pipeline and prediction without using Pipeline.

---

## Author

Abhishek Yadav

B.Tech Information Technology Student
Learning Machine Learning, Data Science, and Python
