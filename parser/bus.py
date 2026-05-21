import pandas as pd
import json
import re

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
EXCEL_FILE = "./data/placement.xlsx"

# ------------------------------------------------------------
# 1. Find the sheet that contains bus route data
# ------------------------------------------------------------
xls = pd.ExcelFile(EXCEL_FILE)
target_sheet = None

# Keywords that appear in bus route tables
route_keywords = ["ashta-palus", "ashta-dudhondi", "ashta-miraj", "karad", "fee"]

for sheet in xls.sheet_names:
    df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
    all_text = " ".join(df_test.fillna('').astype(str).values.flatten()).lower()
    # Check if several route names exist
    if all(kw in all_text for kw in route_keywords[:3]):  # at least first three
        target_sheet = sheet
        break

# Fallback: look for any sheet containing "Sr.No" and a route name pattern
if target_sheet is None:
    for sheet in xls.sheet_names:
        df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
        for idx, row in df_test.iterrows():
            row_text = " ".join([str(c).lower() for c in row if pd.notna(c)])
            if "sr.no" in row_text and ("ashta" in row_text or "palus" in row_text):
                target_sheet = sheet
                break
        if target_sheet:
            break

if target_sheet is None:
    raise ValueError("Could not find bus route data. Available sheets: " + str(xls.sheet_names))

print(f"📌 Using sheet: {target_sheet}")
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=target_sheet, header=None, dtype=str)

# ------------------------------------------------------------
# 2. Locate all route start rows (rows with "Sr.No" in column 0)
# ------------------------------------------------------------
route_start_rows = []
for idx, row in df_raw.iterrows():
    first_cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
    if first_cell == "sr.no":
        route_start_rows.append(idx)

if not route_start_rows:
    raise ValueError("No 'Sr.No' rows found.")
print(f"Found {len(route_start_rows)} route sections at rows: {route_start_rows}")

# ------------------------------------------------------------
# 3. Extract each route
# ------------------------------------------------------------
all_routes = []   # list of (route_name, list of (stop_name, fee))

for i, start_row in enumerate(route_start_rows):
    # The route name is in column 1 of the header row
    header_row = df_raw.iloc[start_row]
    route_name = str(header_row.iloc[1]).strip() if pd.notna(header_row.iloc[1]) else ""
    if not route_name:
        # If merged, maybe route name is in column 0 and "Sr.No" is elsewhere? Skip if missing
        route_name = f"Route_{i+1}"

    # Data rows start from start_row+1 until next "Sr.No" or blank row
    end_row = route_start_rows[i+1] if i+1 < len(route_start_rows) else len(df_raw)
    
    stops = []
    for j in range(start_row + 1, end_row):
        row = df_raw.iloc[j]
        if row.isna().all():
            continue
        stop_name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        fee = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        # Skip rows with empty stop or fee
        if stop_name and fee:
            stops.append((stop_name, fee))
    
    if stops:
        all_routes.append((route_name, stops))
        print(f"   ✅ {route_name}: {len(stops)} stops")

# ------------------------------------------------------------
# 4. Build knowledge sentences
# ------------------------------------------------------------
knowledge_docs = []

for route_name, stops in all_routes:
    # Option 1: per‑stop sentences
    for stop, fee in stops:
        sentence = (f"For the bus route {route_name}, the stop '{stop}' "
                    f"has a monthly fee of ₹{fee}.")
        knowledge_docs.append(sentence)
    
    # Option 2: a summary sentence for the whole route
    stop_list = ", ".join([f"{s} (₹{f})" for s, f in stops])
    summary = (f"The bus route {route_name} covers the following stops with monthly fees: {stop_list}.")
    knowledge_docs.append(summary)

# ------------------------------------------------------------
# 5. Save output
# ------------------------------------------------------------
json_path = "bus_fees_knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved {len(knowledge_docs)} entries to {json_path}")

txt_path = "bus_fees_knowledge.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for doc in knowledge_docs:
        f.write(doc + "\n")
print(f"📄 Text version saved to {txt_path}")

# Show a few samples
if knowledge_docs:
    print("\n🔍 Sample sentences:")
    for i, s in enumerate(knowledge_docs[:5], 1):
        print(f"{i}. {s}")