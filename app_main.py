from flask import Flask, render_template, request, jsonify
import joblib  # For loading a pre-trained model and scaler
import numpy as np

app = Flask(__name__)

# Load your pre-trained ML model and scaler
model = joblib.load(r'C:\Users\Bipin\Downloads\Cardio Vascular Prediction with Deployment\model.pkl','rb')  # Replace with your model file path
scaler = joblib.load('scaler.pkl')  # Replace with your scaler file path

@app.route('/')
def index():
    return render_template('index_main.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = {
        'gender': int(request.form.get('gender')),
        'height': float(request.form.get('height')),
        'weight': float(request.form.get('weight')),
        'ap_hi': int(request.form.get('ap_hi')),
        'ap_lo': int(request.form.get('ap_lo')),
        'cholesterol': int(request.form.get('cholesterol')),
        'gluc': int(request.form.get('gluc')),
        'smoke': int(request.form.get('smoke')),
        'alco': int(request.form.get('alco')),
        'active': int(request.form.get('active')),
        'age_years': int(request.form.get('age_years'))
    }
    
    # Convert data to the format required by the model (e.g., a list or array)
    features = [
        data['gender'],
        data['height'],
        data['weight'],
        data['ap_hi'],
        data['ap_lo'],
        data['cholesterol'],
        data['gluc'],
        data['smoke'],
        data['alco'],
        data['active'],
        data['age_years']
    ]
    
    # Transform the features using the scaler
    features = np.array(features).reshape(1, -1)  # Reshape for scaler
    scaled_features = scaler.transform(features)
    
    # Predict using the ML model
    prediction = model.predict(scaled_features)
    
    # Convert numpy types to standard Python types
    prediction = prediction.tolist()  # Convert numpy array to list
    
    # Return the prediction result as JSON
    result = {'prediction': prediction[0]}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
