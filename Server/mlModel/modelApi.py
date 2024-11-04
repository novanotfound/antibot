from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('one_class_svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Data from the MERN backend
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})
