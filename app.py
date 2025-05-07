from flask import Flask, request, render_template, jsonify  
import numpy as np  
import pandas as pd  
import joblib  

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("trained_emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define valid ranges for each physiological signal
valid_ranges = {
    "heart_rate": (40, 180),  
    "skin_temp": (30, 40),
    "gsr": (0.01, 10),
    "respiration_rate": (8, 40)
}

@app.route('/')
def home():
    return render_template('index.html')  # Renders the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values
        heart_rate = float(request.form['heart_rate'])
        skin_temp = float(request.form['skin_temp'])
        gsr = float(request.form['gsr'])
        respiration_rate = float(request.form['respiration_rate'])

        # Store input values in a dictionary
        input_values = {
            "heart_rate": heart_rate,
            "skin_temp": skin_temp,
            "gsr": gsr,
            "respiration_rate": respiration_rate
        }

        # Validate input ranges
        for key, value in input_values.items():
            min_val, max_val = valid_ranges[key]
            if not (min_val <= value <= max_val):
                return jsonify({'error': f'Invalid {key}. Must be between {min_val} and {max_val}.'})

        # Prepare and scale the input data
        input_data = np.array([[heart_rate, skin_temp, gsr, respiration_rate]])
        scaled_data = scaler.transform(input_data)

        # Feature extraction
        features_df = pd.DataFrame({
            'mean': [np.mean(scaled_data)],
            'std_dev': [np.std(scaled_data)]
        })

        # Make prediction
        predicted_label = model.predict(features_df)[0]
        predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

        return jsonify({'emotion': predicted_emotion})

    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter numerical values.'})

if __name__ == '__main__':
    app.run(debug=True)
