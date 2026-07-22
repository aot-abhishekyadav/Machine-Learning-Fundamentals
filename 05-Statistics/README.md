# 📊 Statistics Fundamentals with Python

This repository contains my practical learning of **Statistics using Python**. It covers descriptive statistics, data visualization, outlier detection, hypothesis testing and correlation using real datasets.

## 📚 Topics Covered

### Descriptive Statistics

* Mean
* Median
* Mode
* Standard Deviation
* Percentiles
* Quartiles
* IQR

### Data Visualization

* Box Plot
* Histogram
* KDE Plot
* Count Plot
* Pair Plot

### Outlier Detection

* Z-Score Method
* IQR Method
* Lower and Upper Fence

### Hypothesis Testing

* Z-Test
* One-Sample T-Test
* Independent Two-Sample T-Test
* Null and Alternative Hypothesis
* P-value and Significance Level

### Correlation

* Correlation Matrix
* Positive and Negative Correlation
* Pairplot for visual analysis

## 🛠️ Libraries Used

```python
numpy
pandas
seaborn
matplotlib
statistics
scipy
statsmodels
```

## 📂 Datasets Used

* Tips Dataset
* Iris Dataset
* Custom Outlier Dataset
* LED Bulb Lifetime Dataset
* Age Dataset
* Manufacturing Process Dataset

## 🔍 Example: Outlier Detection Using Z-Score

```python
def detect_outliers(data):
    outliers = []
    mean = np.mean(data)
    std = np.std(data)

    for value in data:
        z_score = (value - mean) / std

        if abs(z_score) > 3:
            outliers.append(value)

    return outliers
```

## 🧪 Hypothesis Testing Rule

```python
if p_value < 0.05:
    print("Reject the Null Hypothesis")
else:
    print("Fail to Reject the Null Hypothesis")
```

* `p-value < 0.05` → Reject Null Hypothesis
* `p-value ≥ 0.05` → Fail to Reject Null Hypothesis

## 🎯 What I Learned

* How to calculate statistical measures
* How to visualize data distributions
* How to identify outliers
* How to perform Z-tests and T-tests
* How to interpret p-values
* How to calculate and visualize correlation
* How statistics is used in Data Science and Machine Learning

## 👨‍💻 Author

**Abhishek Yadav**

* GitHub: [aot-abhishekyadav](https://github.com/aot-abhishekyadav)

## ⭐ Support

If you found this repository useful, consider giving it a ⭐.
