import requests
import pandas as pd

# Load your cleaned CSV data
df = pd.read_csv('data/market_prices.csv')

# Convert dataframe to list of dictionaries
historical_data = df.to_dict(orient='records')

# Send data to Flask API
response = requests.post(
    "http://127.0.0.1:5000/train",
    json={"historical_data": historical_data}
)

# Print API response
print(response.json())