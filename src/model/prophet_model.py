import pandas as pd
from prophet import Prophet
import joblib
import os

MODEL_DIR = "saved_models"

def train_model(historical_data):
    df = pd.DataFrame(historical_data)
    df = df.rename(columns={'date': 'ds', 'price': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    model = Prophet()
    model.fit(df[['ds', 'y']])
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "prophet_model.pkl")
    joblib.dump(model, model_path)
    return model_path

def predict_prices(future_dates):
    model_path = os.path.join(MODEL_DIR, "prophet_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please train the model first.")
    model = joblib.load(model_path)
    future = pd.DataFrame({'ds': pd.to_datetime(future_dates)})
    forecast = model.predict(future)
    result = []
    for _, row in forecast.iterrows():
        result.append({
            "date": row['ds'].strftime('%Y-%m-%d'),
            "predicted_price": round(row['yhat'], 2),
            "lower": round(row['yhat_lower'], 2),
            "upper": round(row['yhat_upper'], 2)
        })
    return result