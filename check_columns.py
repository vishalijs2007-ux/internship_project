import pandas as pd

# Read dataset
df = pd.read_csv("internship.csv")

# Show all column names
print("Columns in dataset:")
print(df.columns.tolist())

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())