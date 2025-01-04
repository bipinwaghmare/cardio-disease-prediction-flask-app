import pandas as pd
import numpy as np
import joblib
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score
import os
from sklearn.model_selection import train_test_split
import pickle

x = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data\x.csv")
y = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data\y.csv")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

x_train.shape, y_train.shape

x_test.shape, y_test.shape

from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)

print(classification_report(y_test, pred))

print(accuracy_score(y_test,pred))


# Save the model and the scaler
model_path = r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\pkl files\model.pkl'

# Save the model using pickle
with open(model_path, 'wb') as file:
    pickle.dump(xgb, file)