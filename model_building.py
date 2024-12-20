import pandas as pd

# Define paths for the top 3 API files
file_paths = {
    "A9": "/Users/prince/Downloads/A9.csv",
    "A2": "/Users/prince/Downloads/A2.csv",
    "A7": "/Users/prince/Downloads/A7.csv"
}

# Define a function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    # Convert 'Time of call' to datetime
    data['Time of call'] = pd.to_datetime(data['Time of call'], dayfirst=True)
    # Set 'Time of call' as the index for time series analysis
    data.set_index('Time of call', inplace=True)
    # Resample data to aggregate counts by minute, hour, etc., as needed
    data_resampled = data.resample('h').count()
    return data_resampled

# Load and preprocess data for each API
data_A9 = load_data(file_paths["A9"])
data_A2 = load_data(file_paths["A2"])
data_A7 = load_data(file_paths["A7"])

#data pre-processing
# Print sample of resampled data for each API
print("Sample of A9 API data:")
print(data_A9.head())

print("\nSample of A2 API data:")
print(data_A2.head())

print("\nSample of A7 API data:")
print(data_A7.head())


#training models

#model 1 ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Define a function to train an ARIMA model and return the evaluation metric (e.g., MAE)
def train_arima(data):
    model = ARIMA(data, order=(1, 1, 1))  # Adjust order based on data analysis
    model_fit = model.fit()
    return model_fit

#Model 2 exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_exponential_smoothing(data):
    model = ExponentialSmoothing(data, seasonal='add', seasonal_periods=24).fit()
    return model

# Model 3 SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Define a function to train a SARIMA model
def train_sarima(data):
    # Adjust order and seasonal_order based on data analysis
    model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
    model_fit = model.fit(disp=False)
    return model_fit



from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define evaluation function
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(model, data, forecast_steps=10):
    """Evaluates the model by forecasting and comparing with actual data."""
    # Forecast future values for the specified number of steps
    predictions = model.forecast(steps=forecast_steps)
    
    # Extract the last `forecast_steps` values from the actual data
    actual = data[-forecast_steps:]  # Adjust the length to match `predictions`

    # Calculate the Mean Absolute Error and Root Mean Squared Error
    mae = mean_absolute_error(actual, predictions)
    rmse = mean_squared_error(actual, predictions, squared=False)
    
    return mae, rmse


import pickle

# Evaluate all models for each API and store the best model objects
api_models = {"A9": data_A9, "A2": data_A2, "A7": data_A7}
best_models = {}

for api, data in api_models.items():
    results = {}
    
    # Train and evaluate ARIMA model
    arima_model = train_arima(data)
    results['ARIMA'] = (evaluate_model(arima_model, data), arima_model)
    
    # Train and evaluate ETS model
    ets_model = train_exponential_smoothing(data)
    results['ETS'] = (evaluate_model(ets_model, data), ets_model)
    
    # Train and evaluate SARIMA model
    sarima_model = train_sarima(data)
    results['SARIMA'] = (evaluate_model(sarima_model, data), sarima_model)
    
    # Find the model with the lowest MAE for each API
    best_model_name, (best_metrics, best_model) = min(results.items(), key=lambda x: x[1][0][0])  # Min by MAE
    best_models[api] = best_model  # Store the best model object for saving
    print(f"Best model for {api} is {best_model_name} with MAE {best_metrics[0]} and RMSE {best_metrics[1]}")

# Save the best model for each API code to a .pkl file
for api_code, model in best_models.items():
    filename = f'{api_code}_best_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Serialized model for {api_code} saved as {filename}")



    #deployment



