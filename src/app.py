from flask import Flask, request, jsonify
from model.prophet_model import train_model, predict_prices

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    if not data or 'historical_data' not in data:
        return jsonify({"error": "No historical data provided"}), 400
    historical_data = data['historical_data']
    try:
        train_model(historical_data)
        return jsonify({"message": "Model trained successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'future_dates' not in data:
        return jsonify({"error": "No future dates provided"}), 400
    future_dates = data['future_dates']
    try:
        predictions = predict_prices(future_dates)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)