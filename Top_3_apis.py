import pandas as pd

data = pd.read_csv('/Users/prince/Downloads/API Call Dataset.csv', names=['API code', 'Time of call'])

# Step 1: Calculate the frequency of each API code
api_frequency = data['API code'].value_counts()

# Step 2: Identify the top 3 most frequently called APIs
top_3_apis = api_frequency.head(3)

# Display the result
print("Top 3 most called APIs:")
print(top_3_apis)
