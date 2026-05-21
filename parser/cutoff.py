import pandas as pd

# Load cutoff sheet
file_path = "./data/placement.xlsx"

# Read the cutoff sheet
df = pd.read_excel(file_path, sheet_name="cutoff", header=[0, 1])

# Replace '-' with 'Data not found'
df = df.replace("-", "Data not found")

# Optional: also replace empty cells
df = df.fillna("Data not found")

# Flatten multi-level columns
df.columns = [
    f"{col1}_{col2}" if col2 != "Unnamed: 0_level_1" else str(col1)
    for col1, col2 in df.columns
]

# Save cleaned data
df.to_csv("cutoff_cleaned.csv", index=False)

print(df.head())