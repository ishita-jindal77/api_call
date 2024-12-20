import pickle
import pandas as pd
from datetime import datetime, timedelta

# Load models for A9, A2, and A7
model_files = ["A9_best_model.pkl", "A2_best_model.pkl", "A7_best_model.pkl"]
models = {}

# Load each model from its respective file
for model_file in model_files:
    api_code = model_file.split('_')[0]  # Extract API code from filename
    with open(model_file, 'rb') as file:
        models[api_code] = pickle.load(file)
    print(f"Loaded model for {api_code}")

# Function to make a forecast
def make_forecast(api_code, forecast_steps=10):
    """Makes forecast for the specified API code and number of steps."""
    model = models.get(api_code)
    if not model:
        print(f"Model for API {api_code} not found.")
        return None

    # Generate forecast
    predictions = model.forecast(steps=forecast_steps)
    
    # Create a timestamp index for the forecasted values
    last_time = datetime.now()  # Use current time or the last data timestamp
    forecast_times = [last_time + timedelta(hours=i) for i in range(1, forecast_steps + 1)]
    forecast_df = pd.DataFrame({"Time": forecast_times, "Forecast": predictions})
    
    return forecast_df

# Example usage
if __name__ == "__main__":
    api_code = input("Enter the API code (A9, A2, or A7) for forecast: ").upper()
    steps = int(input("Enter the number of forecast steps: "))
    
    # Generate forecast
    forecast = make_forecast(api_code, forecast_steps=steps)
    if forecast is not None:
        print(f"Forecast for {api_code} API:")
        print(forecast)
