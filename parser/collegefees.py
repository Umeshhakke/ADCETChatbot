import pandas as pd

# Read the cutoff sheet
df = pd.read_excel(
    "./data/placement.xlsx",
    sheet_name="fees"
)

# Rename columns properly
df.columns = ["category", "FY", "DSE"]

# Remove fully empty rows
df = df.dropna(how='all')

# Clean numeric columns
df["FY"] = (
    df["FY"]
    .astype(str)
    .str.replace(r'[^0-9]', '', regex=True)
    .astype(int)
)

df["DSE"] = (
    df["DSE"]
    .astype(str)
    .str.replace(r'[^0-9]', '', regex=True)
    .astype(int)
)

# Convert into dictionary
cutoff_data = df.to_dict(orient="records")

# Print parsed data
print(cutoff_data)

# Save JSON
df.to_json(
    "fees.json",
    orient="records",
    indent=4
)

print("\nSaved cutoff_data.json")