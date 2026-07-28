# 🔄 Categorical Data Encoding using Scikit-learn

This project demonstrates how to preprocess categorical data using different encoding techniques in Python with Scikit-learn.

The notebook explains **when to use Ordinal Encoding, One-Hot Encoding, and Label Encoding**, and implements them on a real customer dataset.

---

## 📌 Project Overview

Machine Learning models cannot work directly with categorical (text) data. Therefore, categorical features must be converted into numerical values.

This project covers three major encoding techniques:

- ✅ Ordinal Encoding
- ✅ One-Hot Encoding
- ✅ Label Encoding

It also explains **why ColumnTransformer is useful** when applying different encoding techniques to different columns.

---

## 📂 Dataset

The dataset contains customer information with the following columns:

| Column | Type | Description |
|---------|------|-------------|
| Age | Numerical | Customer's age |
| Gender | Categorical (Nominal) | Male/Female |
| Review | Categorical (Ordinal) | Poor, Average, Good |
| Education | Categorical (Ordinal) | School, UG, PG |
| Purchased | Target Variable | Yes/No |

---

## 🧠 Choosing the Right Encoding

| Situation | Encoding Technique |
|-----------|-------------------|
| Categories have a natural order | ✅ Ordinal Encoding |
| Categories have no order | ✅ One-Hot Encoding |
| Target variable (binary/categorical) | ✅ Label Encoding |

---

## 🔹 Ordinal Encoding

Used for features that have a meaningful order.

### Review

```
Poor < Average < Good
```

Encoded as:

```
Poor    → 0
Average → 1
Good    → 2
```

### Education

```
School < UG < PG
```

Encoded as:

```
School → 0
UG     → 1
PG     → 2
```

Implemented using:

```python
OrdinalEncoder(categories=[
    ['Poor','Average','Good'],
    ['School','UG','PG']
])
```

---

## 🔹 Label Encoding

The target variable **Purchased** contains only two values:

```
No
Yes
```

Encoded as:

```
No  → 0
Yes → 1
```

Implemented using:

```python
LabelEncoder()
```

---

## 🔹 One-Hot Encoding

The **Gender** column is a **Nominal Feature**.

```
Male
Female
```

Since there is **no natural order**, One-Hot Encoding should be used.

Example:

| Gender | Male | Female |
|---------|------|--------|
| Male | 1 | 0 |
| Female | 0 | 1 |

---

## ⚠️ Why ColumnTransformer?

If a dataset contains multiple types of categorical columns, applying different encoders manually becomes tedious.

Without `ColumnTransformer`, we need to:

- Separate columns manually
- Apply different encoders
- Join transformed columns again
- Maintain column order

`ColumnTransformer` automates this entire process and makes preprocessing cleaner and scalable.

---

## 🛠 Libraries Used

- NumPy
- Pandas
- Scikit-learn

Modules:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
```

---

## 📁 Project Workflow

1. Load Dataset
2. Select required columns
3. Split dataset into Train and Test sets
4. Apply Ordinal Encoding on:
   - Review
   - Education
5. Apply Label Encoding on:
   - Purchased
6. (Concept) Apply One-Hot Encoding on:
   - Gender
7. Train machine learning model (next step)

---

## 📊 Encoding Summary

| Feature | Data Type | Encoding |
|----------|-----------|-----------|
| Review | Ordinal | Ordinal Encoder |
| Education | Ordinal | Ordinal Encoder |
| Gender | Nominal | One-Hot Encoder |
| Purchased | Target | Label Encoder |

---

## 🚀 Learning Outcomes

After completing this project, you will understand:

- Difference between Nominal and Ordinal data
- When to use Label Encoding
- When to use Ordinal Encoding
- When to use One-Hot Encoding
- How to preprocess categorical data correctly
- Why ColumnTransformer is preferred in real-world ML pipelines

---

## 📌 Future Improvements

- Implement One-Hot Encoding using `ColumnTransformer`
- Build a complete preprocessing pipeline using `Pipeline`
- Train a machine learning model after preprocessing
- Compare model performance with different encoding techniques

---

## 💻 Author

**Abhishek Yadav**

If you found this project helpful, consider giving it a ⭐ on GitHub.
