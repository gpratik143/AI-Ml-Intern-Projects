# 🚢 Titanic Dataset Preprocessing – README

## 📌 Project Overview

This notebook demonstrates **end-to-end data preprocessing** techniques applied to the Titanic dataset. The goal is to prepare the data for machine learning models by cleaning, transforming, and scaling it. The following key steps were implemented:

---

## 🔧 Steps Performed

### 1. **Importing the Dataset**
- Loaded the Titanic dataset using `pandas`.
- Explored basic info: number of rows/columns, data types, missing values, and summary statistics.

### 2. **Handling Missing Values**
- Used **median imputation** for numerical features.
- Used **mode imputation** for categorical features.
- Ensured no missing values remained in the dataset.

### 3. **Encoding Categorical Variables**
- Converted all categorical columns into numeric format using **One-Hot Encoding**.
- Used `pd.get_dummies()` to prevent introducing any ordinal relationship between categories.

### 4. **Feature Scaling**
- Applied **Standardization** using `StandardScaler` from `sklearn`.
- Scaled all numerical features to have mean = 0 and standard deviation = 1.

### 5. **Outlier Detection and Removal**
- Visualized outliers using **boxplots** for each numeric column.
- Used the **Interquartile Range (IQR)** method to remove outliers from the dataset.
- Replotted boxplots to confirm successful removal.

---

## 📈 Libraries Used

- `pandas` and `numpy` for data manipulation  
- `matplotlib` and `seaborn` for data visualization  
- `sklearn.preprocessing` for feature scaling  

---

## ✅ Outcome

- Dataset cleaned, encoded, and scaled.
- Ready for use in any **machine learning model** (classification, regression, etc.).
- Improved quality and consistency of features.

---

## 📁 File Structure

