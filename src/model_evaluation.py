import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


x = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data\x.csv")
y = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data\y.csv")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

model = joblib.load(r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\pkl files\model.pkl')

y_pred = model.predict(x_test)

y_pred_prob = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Visualize Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc="lower right")
plt.show()

