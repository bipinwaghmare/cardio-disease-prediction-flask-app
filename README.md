# 🏥 **Cardio Prediction Flask App**  

## 🚑 **Cardiovascular Disease Prediction**  

## 📊 **Project Overview**  
This project aims to **predict the risk of cardiovascular disease** using **machine learning techniques**. The dataset, sourced from **Kaggle**, contains **70,000 samples** with **13 features**. The solution involves **data preprocessing, EDA, feature engineering, and classification modeling** to achieve high accuracy.  

---

## 📁 **Dataset**  
- **📌 Source:** Cardiovascular Disease Detection - Kaggle  
- **📏 Shape:** (70000, 13)  
- **🎯 Target Variable:** Cardiovascular Disease (**Binary Classification - 0 or 1**)  

---

## 🔧 **Project Workflow**  

### 🛠️ 1. Data Preprocessing  
✅ Handled **missing values, duplicates, and outliers**  
✅ Performed **data cleaning and normalization**  
✅ **SMOTE** applied to balance the dataset  

### 📊 2. Exploratory Data Analysis (EDA)  
📌 **Pairplots** to visualize feature relationships  
📌 Identified **feature distributions** and **correlations**  
📌 Split data into **X (features) & y (target)**  

### 🔍 3. Feature Engineering  
📌 **Converted age** from days to years  
📌 Created **BMI (Body Mass Index)** column  
📌 Applied **StandardScaler** for feature scaling  

### 🤖 4. Machine Learning Models  
Implemented multiple models for **risk prediction**:  
🔹 **Logistic Regression**  
🔹 **Gradient Boosting Classifier**  
🔹 **Random Forest Classifier**  
🔹 **Decision Tree Classifier**  
🔹 **XGBoost Classifier**  

---

## 📈 **Results**  

### **📊 Model Comparison: Before Feature Engineering & Scaling**  

| 🏆 Model | 🎯 Precision | 🔄 Recall | 📊 F1-Score | ✅ Accuracy |
|---------|------------|---------|-----------|-----------|
| 🔹 Logistic Regression | 0.70 | 0.70 | 0.70 | 70% |
| 🚀 Gradient Boosting | 0.74 | 0.74 | 0.74 | 74% |
| 🌳 Random Forest | 0.72 | 0.72 | 0.72 | 72% |
| 🌱 Decision Tree | 0.63 | 0.63 | 0.63 | 63% |
| ⚡ XGBoost | 0.74 | 0.74 | 0.74 | 74% |

### **📊 Model Comparison: After Feature Engineering & Scaling**  

| 🏆 Model | 🎯 Precision | 🔄 Recall | 📊 F1-Score | ✅ Accuracy |
|---------|------------|---------|-----------|-----------|
| 🔹 Logistic Regression | 0.73 | 0.72 | 0.72 | 72% |
| 🚀 Gradient Boosting | 0.74 | 0.74 | 0.74 | 74% |
| 🌳 Random Forest | 0.72 | 0.72 | 0.72 | 72% |
| 🌱 Decision Tree | 0.64 | 0.64 | 0.64 | 64% |
| ⚡ XGBoost | 0.74 | 0.74 | 0.74 | 74% |

### **🏆 Final Model: XGBoost Classifier**  
After **hyperparameter tuning**, **XGBoost** achieved the best performance:  
✅ **Test Set Accuracy:** **74%**  
✅ **Precision:** **0.75**  
✅ **Recall:** **0.74**  
✅ **F1-Score:** **0.74**  

---

## 🧠 **Key Insights**  
📌 **Feature Engineering** (age conversion & BMI addition) **boosted model performance**  
📌 **XGBoost Classifier outperformed all models** with **74% accuracy**  
📌 **SMOTE** helped balance the dataset and improve prediction for both classes  

---

## 🛠️ **Technologies Used**  
🔹 **Python** – Data preprocessing & modeling  
🔹 **Pandas, NumPy** – Data manipulation  
🔹 **Scikit-Learn** – ML model implementation  
🔹 **XGBoost** – Final model optimization  
🔹 **Flask** – Web app for deployment  
🔹 **HTML/CSS** – Web interface design  

---

## 🚀 **Future Improvements**  
🔹 **Deep Learning** – Implement **Neural Networks** (e.g., LSTMs, CNNs)  
🔹 **Real-Time Predictions** – Deploy the model with an **API**  
🔹 **Hyperparameter Optimization** – Further **fine-tune models**  

---

## Flast App Look
![flask_app_page](https://github.com/user-attachments/assets/38d49885-5e2b-4497-8b23-60ab6e100f17)

---

## 🙋‍♂️ **Author**  
👤 **Bipin Waghmare**  
💻 **Data Scientist & ML Enthusiast**  
🔗 **[GitHub](#) | [LinkedIn](#)**  

---

## 📝 **License**  
This project is licensed under the **MIT License**.  

---

## 🎖 **Acknowledgments**  
Special thanks to **Kaggle & open-source contributors** for providing the dataset.  
