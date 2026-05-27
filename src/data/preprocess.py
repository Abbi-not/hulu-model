import pandas as pd

def load_data(file_path):
    # Support both CSV and Excel
    if file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        data = pd.read_csv(file_path)
    return data

def clean_and_format_data(data):
    # Rename columns to standard names
    data = data.rename(columns={
        'period_date': 'date',
        'product': 'crop_name',
        'market': 'market',
        'value': 'price'
    })
    # Keep only relevant columns
    data = data[['date', 'crop_name', 'market', 'price']]
    # Drop duplicates and missing values
    data = data.drop_duplicates()
    data = data.dropna(subset=['date', 'crop_name', 'market', 'price'])
    # Format date and price
    data['date'] = pd.to_datetime(data['date'])
    data['price'] = pd.to_numeric(data['price'], errors='coerce')
    data = data.sort_values('date')
    return data

def preprocess_and_save(input_path, output_path):
    data = load_data(input_path)
    data = clean_and_format_data(data)
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # Update the input path to your Excel file location
    preprocess_and_save(
        "c:/Users/HP/Downloads/FEWS_NET_Staple_Food_Price_Data (1).xlsx",
        "src/data/market_prices.csv"
    )