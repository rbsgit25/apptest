# # app.py
# from flask import Flask
# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "<h1>Hello from Flask on Heroku!</h1>"

# if __name__ == '__main__':
#     app.run()


from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS  # To allow requests from Angular

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "Text is required"}), 400

        # Transform text and make prediction
        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
