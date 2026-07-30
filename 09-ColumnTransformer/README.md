# 🦠 COVID-19 Data Preprocessing using ColumnTransformer

This project demonstrates how to preprocess a real-world style COVID-19 dataset using different preprocessing techniques from **Scikit-learn**.

The main objective is to understand how to handle:

* Missing values using `SimpleImputer`
* Ordinal categorical features using `OrdinalEncoder`
* Nominal categorical features using `OneHotEncoder`
* Multiple preprocessing operations together using `ColumnTransformer`
* Train-test splitting without data leakage

---

## 📌 Project Overview

The dataset contains information about patients such as:

| Feature     | Description                 | Data Type           |
| ----------- | --------------------------- | ------------------- |
| `age`       | Patient's age               | Numerical           |
| `gender`    | Patient's gender            | Categorical         |
| `fever`     | Patient's fever temperature | Numerical           |
| `cough`     | Severity of cough           | Ordinal Categorical |
| `city`      | Patient's city              | Categorical         |
| `has_covid` | COVID-19 status             | Target              |

The dataset contains **100 records** and **6 columns**.

---

## 🎯 Objective

The objective is to preprocess the dataset before applying a Machine Learning algorithm.

Different columns require different preprocessing techniques:

* `fever` → Missing values → **SimpleImputer**
* `cough` → Ordered categories → **OrdinalEncoder**
* `gender` → Categorical → **OneHotEncoder**
* `city` → Categorical → **OneHotEncoder**
* `age` → Numerical → **Passthrough**

---

## 🗂️ Dataset

Dataset file:

```text
covid_toy.csv
```

Example data:

| age | gender | fever | cough | city    | has_covid |
| --: | ------ | ----: | ----- | ------- | --------- |
|  60 | Male   |   103 | Mild  | Kolkata | No        |
|  27 | Male   |   100 | Mild  | Delhi   | Yes       |
|  42 | Male   |   101 | Mild  | Delhi   | No        |
|  31 | Female |    98 | Mild  | Kolkata | No        |
|  65 | Female |   101 | Mild  | Mumbai  | No        |

---

# 🔎 Data Analysis

## Checking Missing Values

The dataset contains missing values in the `fever` column.

```python
df.isnull().sum()
```

Output:

```text
age           0
gender        0
fever        10
cough         0
city          0
has_covid     0
```

There are **10 missing values** in the `fever` column.

---

# ✂️ Train-Test Split

First, the dataset is divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['has_covid']),
    df['has_covid'],
    test_size=0.2
)
```

This creates:

```text
Training data → 80 rows
Testing data  → 20 rows
```

---

# 🧹 1. SimpleImputer

The `fever` column contains missing values.

We use `SimpleImputer` to fill those missing values.

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer()

X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.transform(X_test[['fever']])
```

### Important

We use:

```python
fit_transform()
```

on training data.

But only:

```python
transform()
```

on test data.

This prevents **data leakage**.

---

# 🔢 2. OrdinalEncoder

The `cough` column contains:

```text
Mild
Strong
```

There is an order between these categories, so we use `OrdinalEncoder`.

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(
    categories=[['Mild', 'Strong']]
)

X_train_cough = oe.fit_transform(X_train[['cough']])
X_test_cough = oe.transform(X_test[['cough']])
```

The encoding becomes:

```text
Mild   → 0
Strong → 1
```

---

# 🏷️ 3. OneHotEncoder

`gender` and `city` are nominal categorical features.

Therefore, we use `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(
    drop='first',
    dtype=np.int32,
    sparse_output=False
)

X_train_gender_city = ohe.fit_transform(
    X_train[['gender', 'city']]
)

X_test_gender_city = ohe.transform(
    X_test[['gender', 'city']]
)
```

### Why `drop='first'`?

It removes one category from each feature to avoid unnecessary redundancy and, for models where it matters, reduce multicollinearity.

For example:

```text
gender:
Female
Male
```

One category can be represented using the other.

---

# 😵 Without ColumnTransformer

If we don't use `ColumnTransformer`, every transformation needs to be handled separately.

### Fever

```python
si = SimpleImputer()

X_train_fever = si.fit_transform(
    X_train[['fever']]
)

X_test_fever = si.transform(
    X_test[['fever']]
)
```

### Cough

```python
oe = OrdinalEncoder(
    categories=[['Mild', 'Strong']]
)

X_train_cough = oe.fit_transform(
    X_train[['cough']]
)

X_test_cough = oe.transform(
    X_test[['cough']]
)
```

### Gender + City

```python
ohe = OneHotEncoder(
    drop='first',
    dtype=np.int32,
    sparse_output=False
)

X_train_gender_city = ohe.fit_transform(
    X_train[['gender', 'city']]
)

X_test_gender_city = ohe.transform(
    X_test[['gender', 'city']]
)
```

Then we manually extract `age`:

```python
X_train_age = X_train.drop(
    columns=['gender', 'fever', 'cough', 'city']
).values

X_test_age = X_test.drop(
    columns=['gender', 'fever', 'cough', 'city']
).values
```

Finally, we concatenate everything:

```python
X_train_new = np.concatenate(
    (
        X_train_age,
        X_train_fever,
        X_train_gender_city,
        X_train_cough
    ),
    axis=1
)

X_test_new = np.concatenate(
    (
        X_test_age,
        X_test_fever,
        X_test_gender_city,
        X_test_cough
    ),
    axis=1
)
```

Result:

```text
X_train_new.shape
(80, 7)
```

This approach works, but it becomes difficult to maintain when the number of features increases.

---

# 🚀 Using ColumnTransformer

`ColumnTransformer` allows us to apply different preprocessing techniques to different columns in a single transformer.

```python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(
    transformers=[
        (
            'tnf1',
            SimpleImputer(),
            ['fever']
        ),
        (
            'tnf2',
            OrdinalEncoder(
                categories=[['Mild', 'Strong']]
            ),
            ['cough']
        ),
        (
            'tnf3',
            OneHotEncoder(
                drop='first',
                dtype=np.int32,
                sparse_output=False
            ),
            ['gender', 'city']
        )
    ],
    remainder='passthrough'
)
```

### Applying the Transformer

```python
X_train_new = transformer.fit_transform(X_train)

X_test_new = transformer.transform(X_test)
```

Output:

```text
X_train_new.shape
(80, 7)

X_test_new.shape
(20, 7)
```

---

# 🔄 How ColumnTransformer Works

The transformer performs the following operations:

```text
                    Original Dataset
                           │
                           ▼
                ┌─────────────────────┐
                │  ColumnTransformer  │
                └─────────────────────┘
                   │       │       │
                   ▼       ▼       ▼
                fever   cough   gender + city
                   │       │       │
                   ▼       ▼       ▼
              Imputer  Ordinal   One-Hot
                       Encoder    Encoder
                   │       │       │
                   └───────┼───────┘
                           │
                           ▼
                    age → passthrough
                           │
                           ▼
                 Final Preprocessed Data
```

---

# 📊 Final Feature Count

After preprocessing, we get:

```text
age               → 1 feature
fever             → 1 feature
cough             → 1 feature
gender            → 1 feature
city              → 3 features
--------------------------------
Total             → 7 features
```

Therefore:

```python
X_train_new.shape
```

returns:

```text
(80, 7)
```

---

# ⚖️ Without vs With ColumnTransformer

| Without ColumnTransformer              | With ColumnTransformer               |
| -------------------------------------- | ------------------------------------ |
| Multiple transformers handled manually | All transformations managed together |
| Need manual concatenation              | No manual concatenation              |
| More code                              | Less code                            |
| More chances of mistakes               | Cleaner workflow                     |
| Harder to maintain                     | Easier to maintain                   |
| Less scalable                          | More scalable                        |

---

# 💡 Why ColumnTransformer is Important

In real-world Machine Learning datasets, different columns often require different preprocessing.

For example:

```text
Numerical → Imputation / Scaling
Categorical → One-Hot Encoding
Ordinal → Ordinal Encoding
Text → Text Vectorization
```

Instead of manually processing every column, `ColumnTransformer` allows us to define all transformations in one place.

It also becomes especially useful when combined with a **Pipeline**.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Google Colab

---



# 🚀 Future Improvements

This preprocessing project can be extended by:

* Adding a Machine Learning classification model
* Creating a complete `Pipeline`
* Adding `StandardScaler`
* Comparing multiple classification algorithms
* Evaluating model performance
* Adding cross-validation
* Hyperparameter tuning
* Deploying the model using Flask or Streamlit

---

# 📚 Key Concepts Learned

Through this project, I learned:

* Handling missing values
* `SimpleImputer`
* Ordinal Encoding
* One-Hot Encoding
* `ColumnTransformer`
* Train-Test Split
* Avoiding Data Leakage
* Feature transformation
* Numerical and categorical feature preprocessing
* Difference between manual preprocessing and automated preprocessing

---

## 👨‍💻 Author

**Abhishek Yadav**

B.Tech Information Technology Student | Aspiring Data Scientist

---

⭐ If you found this project useful, consider giving it a star!
