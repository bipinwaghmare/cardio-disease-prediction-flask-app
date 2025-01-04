import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


df = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data\df.csv")

x = df.drop(['cardio'], axis = 1)
y = df['cardio']


xcols = x.columns
sc = StandardScaler()
x_transformed=sc.fit_transform(x)
x = pd.DataFrame(x_transformed,columns=xcols)
x.head()

folder = r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data"

x.to_csv(os.path.join(folder, 'x.csv'), index=False)
y.to_csv(os.path.join(folder, 'y.csv'), index=False)

print("Data saved successfully to the specified folders!")

joblib.dump(sc, r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\pkl files\scaler.pkl')

