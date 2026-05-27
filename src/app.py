"""
Hulu Farm – Crop Price Prediction Service
==========================================
Flask micro-service that exposes Prophet-based 30-day price predictions.

Endpoints
---------
GET  /predict?days=30
     Returns a list of daily predictions for the next N days (max 90).
     Each item: { date, yhat, lower, upper }

GET  /health
     Returns { status: "ok" }

Run
---
    pip install flask flask-cors prophet pandas
    python app.py

The service listens on port 5001 by default (configurable via PORT env var)
so it doesn't conflict with Django on 8000.
"""

import os
from datetime import date, timedelta

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

# Prophet is an optional heavy dependency – import lazily so the service can
# still start and return fallback data if it's not installed.
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # allow requests from the Expo dev server / production frontend

# ---------------------------------------------------------------------------
# Training data
# Prices in Ethiopian Birr per quintal, weekly averages (simplified).
# Extend this with real CSO / ECX data for production use.
# ---------------------------------------------------------------------------
TRAINING_DATA = {
    "Maize": [
        ("2023-01-01", 2100), ("2023-02-01", 2150), ("2023-03-01", 2200),
        ("2023-04-01", 2180), ("2023-05-01", 2250), ("2023-06-01", 2300),
        ("2023-07-01", 2280), ("2023-08-01", 2320), ("2023-09-01", 2380),
        ("2023-10-01", 2400), ("2023-11-01", 2420), ("2023-12-01", 2450),
        ("2024-01-01", 2430), ("2024-02-01", 2460), ("2024-03-01", 2500),
    ],
    "Wheat": [
        ("2023-01-01", 2800), ("2023-02-01", 2850), ("2023-03-01", 2900),
        ("2023-04-01", 2880), ("2023-05-01", 2950), ("2023-06-01", 3000),
        ("2023-07-01", 2980), ("2023-08-01", 3020), ("2023-09-01", 3050),
        ("2023-10-01", 3080), ("2023-11-01", 3100), ("2023-12-01", 3120),
        ("2024-01-01", 3090), ("2024-02-01", 3110), ("2024-03-01", 3100),
    ],
    "Teff": [
        ("2023-01-01", 4500), ("2023-02-01", 4600), ("2023-03-01", 4700),
        ("2023-04-01", 4680), ("2023-05-01", 4750), ("2023-06-01", 4900),
        ("2023-07-01", 4850), ("2023-08-01", 4950), ("2023-09-01", 5000),
        ("2023-10-01", 5100), ("2023-11-01", 5150), ("2023-12-01", 5200),
        ("2024-01-01", 5180), ("2024-02-01", 5220), ("2024-03-01", 5200),
    ],
    "Beans": [
        ("2023-01-01", 4200), ("2023-02-01", 4250), ("2023-03-01", 4300),
        ("2023-04-01", 4350), ("2023-05-01", 4400), ("2023-06-01", 4450),
        ("2023-07-01", 4500), ("2023-08-01", 4550), ("2023-09-01", 4600),
        ("2023-10-01", 4650), ("2023-11-01", 4700), ("2023-12-01", 4750),
        ("2024-01-01", 4780), ("2024-02-01", 4800), ("2024-03-01", 4800),
    ],
    "Barley": [
        ("2023-01-01", 1900), ("2023-02-01", 1920), ("2023-03-01", 1950),
        ("2023-04-01", 1940), ("2023-05-01", 1960), ("2023-06-01", 2000),
        ("2023-07-01", 1980), ("2023-08-01", 2020), ("2023-09-01", 2050),
        ("2023-10-01", 2060), ("2023-11-01", 2080), ("2023-12-01", 2100),
        ("2024-01-01", 2090), ("2024-02-01", 2095), ("2024-03-01", 2100),
    ],
    "Sorghum": [
        ("2023-01-01", 1700), ("2023-02-01", 1730), ("2023-03-01", 1750),
        ("2023-04-01", 1760), ("2023-05-01", 1800), ("2023-06-01", 1840),
        ("2023-07-01", 1820), ("2023-08-01", 1860), ("2023-09-01", 1900),
        ("2023-10-01", 1920), ("2023-11-01", 1940), ("2023-12-01", 1950),
        ("2024-01-01", 1940), ("2024-02-01", 1945), ("2024-03-01", 1950),
    ],
}

# Cache fitted models so we only train once per process startup
_model_cache: dict = {}


def _get_model(commodity: str):
    """Return (or train and cache) a Prophet model for a given commodity."""
    if commodity in _model_cache:
        return _model_cache[commodity]

    rows = TRAINING_DATA.get(commodity)
    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["ds", "y"])
    df["ds"] = pd.to_datetime(df["ds"])

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,   # 80 % confidence interval → yhat_lower / yhat_upper
        changepoint_prior_scale=0.1,
    )
    m.fit(df)
    _model_cache[commodity] = m
    return m


def _static_fallback(days: int, commodity: str) -> list:
    """
    Return deterministic ±5 % / +8 % range when Prophet is unavailable.
    """
    rows = TRAINING_DATA.get(commodity, [])
    base = rows[-1][1] if rows else 2000
    today = date.today()
    results = []
    for i in range(days):
        d = today + timedelta(days=i)
        results.append({
            "date":  d.isoformat(),
            "yhat":  round(base * 1.02, 2),
            "lower": round(base * 0.95, 2),
            "upper": round(base * 1.08, 2),
        })
    return results


@app.route("/health")
def health():
    return jsonify({"status": "ok", "prophet": PROPHET_AVAILABLE})


@app.route("/predict")
def predict():
    """
    GET /predict?days=30&commodity=Teff

    If 'commodity' is omitted, returns a single series that the frontend
    distributes across all 6 crops (legacy behaviour kept for backward compat).
    """
    try:
        days = min(int(request.args.get("days", 30)), 90)
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400

    commodity = request.args.get("commodity", "Teff")  # default legacy crop

    if not PROPHET_AVAILABLE:
        return jsonify(_static_fallback(days, commodity))

    model = _get_model(commodity)
    if model is None:
        return jsonify(_static_fallback(days, commodity))

    future = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)

    # Return only the future dates (not the training period)
    today_str = date.today().isoformat()
    future_rows = forecast[forecast["ds"] >= today_str].tail(days)

    results = [
        {
            "date":  row["ds"].strftime("%Y-%m-%d"),
            "yhat":  round(float(row["yhat"]), 2),
            "lower": round(float(row["yhat_lower"]), 2),
            "upper": round(float(row["yhat_upper"]), 2),
        }
        for _, row in future_rows.iterrows()
    ]
    return jsonify(results)


@app.route("/predict/all")
def predict_all():
    """
    GET /predict/all?days=30

    Returns predictions for all 6 crops keyed by commodity name.
    Shape: { "Maize": [...], "Wheat": [...], ... }
    """
    try:
        days = min(int(request.args.get("days", 30)), 90)
    except ValueError:
        return jsonify({"error": "days must be an integer"}), 400

    result = {}
    for crop in TRAINING_DATA:
        if not PROPHET_AVAILABLE:
            result[crop] = _static_fallback(days, crop)
            continue
        model = _get_model(crop)
        if model is None:
            result[crop] = _static_fallback(days, crop)
            continue
        future = model.make_future_dataframe(periods=days, freq="D")
        forecast = model.predict(future)
        today_str = date.today().isoformat()
        future_rows = forecast[forecast["ds"] >= today_str].tail(days)
        result[crop] = [
            {
                "date":  row["ds"].strftime("%Y-%m-%d"),
                "yhat":  round(float(row["yhat"]), 2),
                "lower": round(float(row["yhat_lower"]), 2),
                "upper": round(float(row["yhat_upper"]), 2),
            }
            for _, row in future_rows.iterrows()
        ]
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🌾 Hulu Farm prediction service starting on port {port}")
    print(f"   Prophet available: {PROPHET_AVAILABLE}")
    if not PROPHET_AVAILABLE:
        print("   ⚠ Install prophet for real ML predictions: pip install prophet")
    app.run(host="0.0.0.0", port=port, debug=False)