from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model
model = joblib.load('random_forest_model2.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return "Disease Prediction API Working"

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'error': 'Input data must be in JSON format'}), 400

    data = request.json

    # Get the expected features from the model
    expected_features = model.feature_names_in_

    # Ensure the input data has exactly the expected features
    # Add missing features and set them to 0
    input_data = {feature: data.get(feature, 0) for feature in expected_features}

    # Convert input_data to a numpy array in the correct order
    input_query = np.array([list(input_data.values())])

    try:
        # Predict the result
        result = model.predict(input_query)[0]
    except Exception as e:
        return jsonify({'error': f'Error in prediction: {str(e)}'}), 500

    return jsonify({'predicted_disease': str(result)})


if __name__ =='__main__':
    app.run(debug=False,host='0.0.0.0')
