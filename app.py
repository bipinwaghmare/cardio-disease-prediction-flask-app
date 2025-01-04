from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model and scaler using joblib with error handling
model_path = r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\pkl files\model.pkl'
scaler_path = r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\pkl files\scaler.pkl'

def load_joblib_file(filepath):
    try:
        return joblib.load(filepath)
    except EOFError:
        raise EOFError(f"Error loading the file: {filepath}. The file may be corrupted or incomplete.")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the file: {filepath}. Error: {str(e)}")

model = load_joblib_file(model_path)
scaler = load_joblib_file(scaler_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the form
        id = int(request.form['id'])
        age = int(request.form['age'])
        height = int(request.form['height'])
        weight = float(request.form['weight'])
        gender = int(request.form['gender'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # Create a numpy array with the inputs
        data = np.array([[id, age, height, weight, gender, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]])

        # Scale the input data
        scaled_data = scaler.transform(data)

        # Make the prediction
        prediction = model.predict(scaled_data)

        # Convert the prediction to a human-readable format
        prediction_text = 'No Cardiovascular Disease' if prediction[0] == 0 else 'Cardiovascular Disease'

    except Exception as e:
        prediction_text = f'Error in prediction: {str(e)}'

    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
