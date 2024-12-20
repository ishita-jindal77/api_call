import pandas as pd

# Load the dataset
data = pd.read_csv('/Users/prince/Downloads/API Call Dataset.csv', names=['API code', 'Time of call'])

# Calculate the frequency of each API code and get the top 3
top_3_apis = data['API code'].value_counts().head(3).index

# Filter data for each of the top 3 APIs and save to separate CSV files
for api_code in top_3_apis:
    # Filter the data for the current API code
    api_data = data[data['API code'] == api_code]
    
    # Define the filename based on the API code, e.g., A8.csv
    file_name = f'/Users/prince/Downloads/{api_code}.csv'
    
    # Save the filtered data to the CSV file
    api_data.to_csv(file_name, index=False)
    
    print(f"Data for {api_code} saved to {file_name}")
