import pandas as pd

def load_and_preprocess(filepath, save_csv_path=None):
    df = pd.read_excel(filepath)
    df = df.rename(columns={
        'period_date': 'date',
        'product': 'crop_name',
        'market': 'market',
        'value': 'price'
    })
    df = df[['date', 'crop_name', 'market', 'price']]
    df = df.dropna(subset=['date', 'crop_name', 'market', 'price'])
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.drop_duplicates()
    df = df.sort_values('date')
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df

if __name__ == "__main__":
    cleaned_df = load_and_preprocess(
        "c:/Users/HP/Downloads/FEWS_NET_Staple_Food_Price_Data (1).xlsx",
        save_csv_path="data/market_prices.csv"
    )
    print("Cleaned data saved to data/market_prices.csv")