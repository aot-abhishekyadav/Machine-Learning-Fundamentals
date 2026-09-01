# Outlier Removal in Machine Learning

## 📌 Overview

This project demonstrates different techniques for detecting and handling **outliers** in datasets using Python, Pandas, NumPy, Matplotlib, and Seaborn.

Two datasets are used:

* **Placement Dataset** – Used to demonstrate Z-Score based outlier removal for normally distributed data.
* **Weight-Height Dataset** – Used to demonstrate Percentile-based outlier removal.

## 🛠️ Techniques Covered

### 1. Z-Score Method

The **Z-Score method** is used when the data is approximately normally distributed.

For the `cgpa` column:

* Mean = **6.961**
* Standard Deviation = **0.616**
* Outlier boundary = Mean ± 3 × Standard Deviation
* **5 outliers** were identified.

Two approaches were demonstrated:

* **Trimming** – removing the outlier rows.
* **Capping (Winsorization)** – replacing extreme values with the upper/lower limits.

### 2. IQR Method

The **IQR (Interquartile Range)** method is useful for skewed data.

For `placement_exam_marks`:

* Q1 = **17**
* Q3 = **44**
* IQR = **27**
* Upper Limit = **84.5**
* Lower Limit = **-23.5**
* **15 outliers** were identified.

Both **trimming** and **capping** techniques were demonstrated.

### 3. Percentile Method

The Percentile method was demonstrated using the `Height` column.

Selected boundaries:

* Lower limit = **1st percentile = 58.13**
* Upper limit = **99th percentile = 74.79**

Values outside these limits were treated as outliers.

## 📊 Visualization

Boxplots and distribution plots were used to compare the data **before and after outlier treatment**.

> Note: `sns.distplot()` is deprecated in newer Seaborn versions. `sns.histplot()` or `sns.displot()` can be used instead.

## 💻 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Google Colab

## 🎯 Key Learning

This project helps understand:

* What are outliers?
* How to detect outliers
* Z-Score method
* IQR method
* Percentile method
* Trimming
* Capping / Winsorization
* Visualization of outliers before and after treatment
