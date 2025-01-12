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
    # Handle form-data or JSON input
    if request.content_type == 'application/json':
        data = request.json
    elif request.content_type.startswith('multipart/form-data'):
        # Convert form-data into a dictionary
        data = {key: request.form.get(key, 0) for key in model.feature_names_in_}
    else:
        return jsonify({'error': 'Unsupported Content-Type. Use JSON or form-data.'}), 400

    # Get the expected features from the model
    expected_features = model.feature_names_in_

    # Ensure the input data includes all required features
    input_data = {feature: float(data.get(feature, 0)) for feature in expected_features}

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
