# Hulu Farm Crop Price Prediction

This project is designed to predict agricultural crop prices using the Prophet forecasting model. It provides a Flask REST API for easy access to predictions and model training.

## Project Structure

```
hulu-farm-crop-price-prediction
├── src
│   ├── api
│   │   └── app.py          # Main entry point for the Flask REST API
│   ├── model
│   │   ├── prophet_model.py # Implementation of the Prophet forecasting model
│   │   └── utils.py        # Utility functions for model evaluation and management
│   ├── data
│   │   └── preprocess.py    # Data ingestion and preprocessing functions
│   └── types
│       └── __init__.py      # Custom types and interfaces
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd hulu-farm-crop-price-prediction
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.

   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask API:**
   Navigate to the `src/api` directory and run the application:
   ```
   python app.py
   ```

   The API will be available at `http://127.0.0.1:5000`.

## Usage Guidelines

- **Health Check:** 
  - Endpoint: `/health`
  - Method: `GET`
  - Description: Check if the API is running.

- **Make Prediction:**
  - Endpoint: `/predict`
  - Method: `POST`
  - Description: Submit historical crop data to receive price predictions.

- **Train Model:**
  - Endpoint: `/train`
  - Method: `POST`
  - Description: Train the Prophet model on new historical data.

## Features

- Predict future crop prices based on historical data.
- RESTful API for easy integration with other systems.
- Data preprocessing to ensure high-quality input for the model.
- Evaluation metrics to assess model performance.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.