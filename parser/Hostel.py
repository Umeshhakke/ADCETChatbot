import pandas as pd
import re

# Read hostel sheet
df = pd.read_excel(
    "./data/placement.xlsx",
    sheet_name="hostel"
)

# Rename columns
df.columns = ["hostel", "fees"]

# Remove empty rows
df = df.dropna(how='all')

# Clean fees column
def clean_fees(x):

    # Convert to string
    x = str(x)

    # Keep only digits
    x = re.sub(r'[^0-9]', '', x)

    return int(x)

df["fees"] = df["fees"].apply(clean_fees)

# Convert to dictionary
hostel_data = df.to_dict(orient="records")

# Print result
print(hostel_data)

# Save JSON
df.to_json(
    "hostel_data.json",
    orient="records",
    indent=4
)

print("\nSaved hostel_data.json")