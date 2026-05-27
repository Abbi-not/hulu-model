import requests

future_dates = [
    "2026-06-01",
    "2026-06-02",
    "2026-06-03"
]

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"future_dates": future_dates}
)

print(response.json())