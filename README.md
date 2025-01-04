# cardio-prediction-flask-app

## Cardiovascular Disease Prediction 🚑

## Project Overview 📊
This project aims to predict whether a person is at risk of cardiovascular disease or heart disease using machine learning techniques. The dataset is sourced from Kaggle and contains 70,000 samples with 13 features. The solution involves data preprocessing, EDA, feature engineering, and building classification models to identify the risk with high accuracy.

## Dataset 📁
- **Source:** Cardiovascular Disease Detection - Kaggle
- **Shape:** (70000, 13)
- **Target Variable:** Cardiovascular Disease (Binary Classification - 0 or 1)

## Project Workflow 🔧
### 1. Data Preprocessing
Handled missing values, duplicates, and outliers.
Performed data cleaning and normalization.
SMOTE was applied to balance the target variable y.
### 2. Exploratory Data Analysis (EDA)
Generated pairplots to visualize feature relationships.
Identified feature distributions and correlations.
Split data into X (features) and y (target).
### 3. Feature Engineering
Converted age from days to years.
Created a new BMI (Body Mass Index) column.
Applied StandardScaler for feature scaling.
### 4. Machine Learning Models
Multiple models were implemented to predict the risk of cardiovascular disease:
- Logistic Regression
- Gradient Boosting Classifier
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost Classifier

## Results 📈
**Model Comparison: Before Feature Engineering & Scaling**

| Model              | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Logistic Regression| 0.70      | 0.70   | 0.70     | 70%      |
| Gradient Boosting  | 0.74      | 0.74   | 0.74     | 74%      |
| Random Forest      | 0.72      | 0.72   | 0.72     | 72%      |
| Decision Tree      | 0.63      | 0.63   | 0.63     | 63%      |
| XGBoost            | 0.74      | 0.74   | 0.74     | 74%      |

**Model Comparison: After Feature Engineering & Scaling**

| Model              | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Logistic Regression| 0.73      | 0.72   | 0.72     | 72%      |
| Gradient Boosting  | 0.74      | 0.74   | 0.74     | 74%      |
| Random Forest      | 0.72      | 0.72   | 0.72     | 72%      |
| Decision Tree      | 0.64      | 0.64   | 0.64     | 64%      |
| XGBoost            | 0.74      | 0.74   | 0.74     | 74%      |

**Final Model: XGBoost Classifier**

After hyperparameter tuning, the XGBoost model performed best:
- Test Set Accuracy: 74%
- Precision: 0.75
- Recall: 0.74
- F1-Score: 0.74

## Key Insights 🧠
- Feature Engineering (converting age and adding BMI) improved overall model performance.
- XGBoost Classifier outperformed all other models with an accuracy of 74%.
- Balancing the dataset using SMOTE improved predictions for both classes.

## Technologies Used 🛠️
- **Python:** Data preprocessing, model building
- **Pandas, NumPy:** Data manipulation
- **Scikit-Learn:** Machine learning models
- **XGBoost:** Final model implementation
- **Flask:** Web app for deployment
- **HTML/CSS:** Web interface design

## Future Improvements 🚀
- Implement deep learning techniques (e.g., neural networks).
- Add real-time predictions and API integration.
- Optimize hyperparameter tuning further.

## Author 🙋‍♂️
- Bipin Waghmare
- Data Scientist & Machine Learning Enthusiast
- GitHub | LinkedIn
