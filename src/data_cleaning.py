import pandas as pd
import numpy as np
import os


df = pd.read_csv(r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\cardio_train.csv")

print(df)

# df.drop('id', axis = 1, inplace = True)

if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
else:
    print("Column 'id' does not exist in the DataFrame.")

df['weight'] = df['weight'].astype(int)

df['age'] = df['age']/365

df['bmi'] = df['weight']/(df['height']/100)**2

df['age'] = df['age'].astype(int)

folder = r"C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\data"

try:
    df.to_csv(os.path.join(folder, 'df.csv'), index=False)
except:
    print("Error in saving the file")