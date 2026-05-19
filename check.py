import pandas as pd
import json

EXCEL_FILE = "./data/FY CUT OFF  Placement data, Fee, Hostel 2025 - 26.xlsx"

# Check available sheets
xls = pd.ExcelFile(EXCEL_FILE)
print("Available sheets:", xls.sheet_names)
# Use the first sheet if you're not sure which one
SHEET_NAME = xls.sheet_names[0]   # or manually set to the correct name

df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None, dtype=str)