def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def load_dataset(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df