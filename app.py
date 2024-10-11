
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model (replace 'disease_prediction_model.pkl' with your model file path)
model = joblib.load('disease_prediction_model.pkl')

@app.route('/')
def home():
    return "Welcome to COZOM API"

@app.route('/predict', methods=['POST'])
def predict():
    symptom = request.json.get('symptom')
    prediction = model.predict([symptom])[0]
    return jsonify({'disease': prediction})

if __name__ == '__main__':
    app.run(debug=True)
