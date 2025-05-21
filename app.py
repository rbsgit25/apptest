# # app.py
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "<h1>Hello from Flask on Heroku!</h1>"

# if __name__ == '__main__':
#     app.run()


# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib
import numpy as np

app = Flask(__name__)

# Enable CORS for all routes in the app
CORS(app)

# Load the pre-trained model
# model = joblib.load('linear_regression_model.pkl')

model = joblib.load('salary_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.get_json()

        # Assuming the input is a number (e.g., X value for prediction)
        X_input = np.array(data['X']).reshape(-1, 1)

        # Make the prediction
        prediction = model.predict(X_input)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)




# if __name__ == "__main__":
#     import os
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port)
