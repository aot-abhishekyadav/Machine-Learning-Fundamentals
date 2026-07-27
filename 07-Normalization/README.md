# 📊 Feature Scaling in Machine Learning

This project demonstrates the importance of **Feature Scaling** in Machine Learning using the **Wine Dataset**. It covers different scaling techniques and explains when to use each technique based on the Machine Learning algorithm and dataset characteristics.

---

## 📌 Project Overview

In Machine Learning, numerical features can have very different ranges.

For example:

* **Age:** 18 – 60
* **Salary:** ₹20,000 – ₹15,00,000

If features are not scaled, some Machine Learning algorithms may give more importance to features with larger numerical values.

**Feature Scaling** transforms numerical features to a comparable scale so that algorithms can process them more effectively.

This project explores:

* Min-Max Normalization
* Standardization
* Robust Scaling
* MaxAbs Scaling
* Train-Test Split
* Data Visualization
* Distribution Analysis
* Practical implementation using Scikit-learn

---

## 🧠 What is Feature Scaling?

Feature scaling is a preprocessing technique used to bring numerical features to a similar scale.

It is especially important for algorithms that depend on:

* Distance calculations
* Gradient descent
* Variance
* Magnitude of feature values

### Algorithms where scaling is generally important:

* K-Nearest Neighbors (KNN)
* K-Means Clustering
* Support Vector Machines (SVM)
* Principal Component Analysis (PCA)
* Neural Networks
* Logistic Regression
* Linear Regression with regularization
* Ridge Regression
* Lasso Regression

### Algorithms that generally do not require feature scaling:

* Decision Tree
* Random Forest
* XGBoost
* LightGBM
* CatBoost

Tree-based algorithms split data based on feature thresholds, so they are generally not affected by differences in feature scale.

---

# 🔹 1. Min-Max Normalization

Min-Max Scaling transforms features into a fixed range, usually **0 to 1**.

### Formula

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

### Example

Suppose:

```text
Marks = 70
Minimum Marks = 40
Maximum Marks = 100
```

Then:

```text
X_scaled = (70 - 40) / (100 - 40)
         = 30 / 60
         = 0.5
```

Therefore:

```text
70 → 0.5
```

### Python Implementation

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Important

The scaler is fitted **only on the training data**.

```python
scaler.fit(X_train)
```

Then the same transformation is applied to both training and testing data.

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents **Data Leakage**.

---

# 🔹 2. Standardization

Standardization transforms the data so that:

* Mean ≈ 0
* Standard Deviation ≈ 1

### Formula

```text
X_scaled = (X - Mean) / Standard Deviation
```

Standardization is commonly used when features have different scales and the algorithm benefits from centered data.

### Python Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 🔹 3. Robust Scaling

Robust Scaling uses the **Median** and **Interquartile Range (IQR)**.

It is less sensitive to outliers compared to Min-Max Scaling and Standardization.

### Formula

```text
X_scaled = (X - Median) / IQR
```

Where:

```text
IQR = Q3 - Q1
```

* Q1 = 25th Percentile
* Q3 = 75th Percentile

### Python Implementation

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Best Use Case

Use RobustScaler when the dataset contains significant **outliers**.

---

# 🔹 4. MaxAbs Scaling

MaxAbsScaler divides every feature by its maximum absolute value.

### Formula

```text
X_scaled = X / |X_max|
```

The transformed values generally fall between:

```text
-1 and +1
```

It is particularly useful when working with **sparse datasets**, because it does not center the data around zero.

### Python Implementation

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

# 📊 Scaling Techniques Comparison

| Scaling Technique | Main Idea                        | Outlier Sensitivity | Typical Range  |
| ----------------- | -------------------------------- | ------------------- | -------------- |
| MinMaxScaler      | Uses Minimum and Maximum         | High                | 0 to 1         |
| StandardScaler    | Uses Mean and Standard Deviation | Medium              | No fixed range |
| RobustScaler      | Uses Median and IQR              | Low                 | No fixed range |
| MaxAbsScaler      | Uses Maximum Absolute Value      | High                | -1 to +1       |

---

# 🍷 Dataset Used

This project uses the **Wine Dataset**.

The dataset contains multiple chemical properties of wine samples.

For this demonstration, the following features are used:

* **Class Label**
* **Alcohol**
* **Malic Acid**

The project focuses on understanding how scaling changes the numerical range and distribution of features.

---

# 🔬 Project Workflow

The complete workflow followed in this project is:

```text
Load Dataset
      ↓
Select Required Features
      ↓
Rename Columns
      ↓
Exploratory Data Analysis
      ↓
Visualize Feature Distributions
      ↓
Train-Test Split
      ↓
Apply Min-Max Scaling
      ↓
Compare Before vs After Scaling
      ↓
Analyze Feature Distributions
```

---

# 📈 Visualizations

The project includes visual analysis using:

### KDE Plots

Used to understand the distribution of:

* Alcohol
* Malic Acid

### Scatter Plots

Used to visualize the relationship between:

* Alcohol
* Malic Acid
* Class Labels

### Before vs After Scaling

The project compares the feature space before and after applying Min-Max Scaling.

This helps demonstrate that scaling changes the **numerical scale** of the features but does not fundamentally change their relative distribution pattern.

---

# 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Google Colab

---



# 🚀 How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Open the Notebook

You can run the project using:

* Jupyter Notebook
* Google Colab

### 4. Load the Dataset

Update the dataset path according to your environment.

Example:

```python
df = pd.read_csv('/content/drive/MyDrive/wine_data.csv',
                 header=None,
                 usecols=[0, 1, 2])
```

---

# 💡 Key Learnings

Through this project, I learned:

* Why feature scaling is important in Machine Learning
* Difference between Normalization and Standardization
* How Min-Max Scaling works
* How StandardScaler works
* How RobustScaler handles outliers
* How MaxAbsScaler works
* Why scaling should be fitted only on training data
* How to avoid data leakage
* How scaling affects feature distributions
* Which Machine Learning algorithms require feature scaling
* Why tree-based algorithms generally don't require scaling

---

# 📌 Important Note

Feature scaling does **not always improve every Machine Learning model**.

The choice of scaling technique depends on:

* Machine Learning algorithm
* Presence of outliers
* Feature distribution
* Whether the data is sparse
* Whether a fixed range is required

For many Machine Learning workflows, **StandardScaler is a common default choice**, while **RobustScaler** can be a better option when strong outliers are present.

---

# 👨‍💻 Author

**Abhishek Yadav**

B.Tech Information Technology Student | Machine Learning Enthusiast

---

⭐ If you found this project useful, consider giving the repository a **star**!
