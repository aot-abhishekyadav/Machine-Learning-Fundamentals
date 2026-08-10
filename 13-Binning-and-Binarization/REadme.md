# 📊 Binning & Binarization in Machine Learning

This project demonstrates two important **feature engineering techniques** used to transform numerical data:

* **Binning (Discretization)** – Converts continuous numerical values into discrete intervals (bins).
* **Binarization** – Converts numerical values into binary values (`0` or `1`) using a threshold.

## 🎯 Why Use Binning & Binarization?

Although numerical features often perform well, converting them can sometimes:

* Reduce the effect of noise and outliers
* Simplify complex numerical distributions
* Make features easier to interpret
* Improve model performance in certain cases
* Create meaningful categories from continuous values

### Example

Instead of using exact app download counts:

`500 → 1000 → 10000 → 1M`

we can create categories such as:

`100+ | 10K+ | 1M+`

---

## 1️⃣ Binning (Discretization)

Binning divides continuous numerical values into different intervals.

### Types Covered

**1. Equal Width Binning**

* Divides the complete range into equal-sized intervals.
* Simple and fast.
* Can be sensitive to outliers.

**2. Equal Frequency (Quantile) Binning**

* Each bin contains approximately the same number of observations.
* Less affected by outliers.
* Bin widths can be different.

**3. K-Means Binning**

* Uses K-Means clustering to create groups based on similarity between values.

**4. Custom Binning**

* User manually defines meaningful ranges.

Example:

`0–18 → Child`
`19–35 → Adult`
`36–60 → Middle Age`
`60+ → Senior`

### Titanic Dataset Experiment

Binning was applied to **Age** and **Fare** using `KBinsDiscretizer`.

| Strategy |   Accuracy |
| -------- | ---------: |
| Quantile | **0.6401** |
| K-Means  |     0.6345 |
| Uniform  |     0.6261 |

In this experiment, **Quantile Binning performed best** among the tested strategies.

---

## 2️⃣ Binarization

Binarization converts numerical values into only two values: **0 and 1**.

### Formula

```text
x ≤ threshold → 0
x > threshold → 1
```

Example:

```text
Age ≤ 18 → 0
Age > 18 → 1
```

In the Titanic dataset, the `family` feature was binarized to represent whether a passenger had family members (`1`) or not (`0`).

### Result

The experiment showed that binarization **did not improve accuracy** for this particular Decision Tree model.

---

## 🛠️ Libraries Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## 📌 Key Takeaway

**Binning and Binarization are not mandatory transformations.** Their usefulness depends on the dataset, feature distribution, algorithm, and problem.

In these experiments, **Quantile Binning gave the best improvement**, while Binarization slightly reduced the model's performance.

---

## 📂 Topics Covered

```text
Feature Engineering
│
├── Binning / Discretization
│   ├── Equal Width
│   ├── Equal Frequency / Quantile
│   ├── K-Means
│   └── Custom Binning
│
└── Binarization
    └── Threshold-based 0/1 transformation
```
