import pandas as pd
import json
import re

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
EXCEL_FILE = "./data/FY CUT OFF  Placement data, Fee, Hostel 2025 - 26.xlsx"
TARGET_SHEET = "Sheet4"          # <-- Change if needed

# ------------------------------------------------------------
# 1. Load Sheet4
# ------------------------------------------------------------
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=TARGET_SHEET, header=None, dtype=str)
print(f"📌 Using sheet: {TARGET_SHEET}")

# ------------------------------------------------------------
# 2. Find the row with year labels (search the whole sheet)
# ------------------------------------------------------------
year_label_row = None
year_info = []  # list of (column_index, year_string)

for idx, row in df_raw.iterrows():
    cols_with_year = []
    for c, cell in enumerate(row):
        if isinstance(cell, str) and re.search(r"acad[ae]mic year", cell.lower()):
            yr_match = re.search(r"(\d{4}-\d{2,4})", cell)
            if yr_match:
                cols_with_year.append((c, yr_match.group(1)))
    if cols_with_year:
        year_label_row = idx
        year_info = cols_with_year
        break

if year_label_row is None:
    raise ValueError("No row with 'Acadamic Year' labels found.")
print(f"Year labels found in row {year_label_row}:")
for col, yr in year_info:
    print(f"   Column {col} → {yr}")

# ------------------------------------------------------------
# 3. Find header row (should be the next row after year labels)
# ------------------------------------------------------------
header_row = year_label_row + 1
while header_row < len(df_raw) and df_raw.iloc[header_row].isna().all():
    header_row += 1
if header_row >= len(df_raw):
    raise ValueError("No header row found after year labels.")
print(f"Header row is {header_row}")

# (Optional) verify that header row contains "Total Students"
header_text = " ".join([str(c).lower() for c in df_raw.iloc[header_row] if pd.notna(c)])
if "total students" not in header_text:
    print("⚠️ Warning: Header row may not contain 'Total Students' – column order might be wrong.")

# ------------------------------------------------------------
# 4. Extract data rows
# ------------------------------------------------------------
# Branch names are always in column 0.
# For each year column (col_start), the data columns are:
#   col_start+1 : Total Students
#   col_start+2 : Students Placed
#   col_start+3 : Avg. Salary (LPA)
#   col_start+4 : Highest Salary (LPA)

data_rows = []
for i in range(header_row + 1, len(df_raw)):
    row = df_raw.iloc[i]
    if row.isna().all():
        continue
    first_cell = str(row.iloc[0]).strip().lower() if pd.notna(row.iloc[0]) else ""
    if "overall total" in first_cell:
        break

    branch = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
    if not branch:
        continue

    row_data = {"Branch": branch}
    for col_start, yr in year_info:
        # Try to extract the four numbers
        def safe_get(col_offset):
            col = col_start + 1 + col_offset
            if col < len(row) and pd.notna(row.iloc[col]):
                return str(row.iloc[col]).strip()
            return ""
        total = safe_get(0)
        placed = safe_get(1)
        avg = safe_get(2)
        high = safe_get(3)
        # Only store if at least one field is non-empty
        if any([total, placed, avg, high]):
            row_data[yr] = {
                "Total Students": total,
                "Students Placed": placed,
                "Average Salary (LPA)": avg,
                "Highest Salary (LPA)": high
            }
    data_rows.append(row_data)

print(f"Extracted {len(data_rows)} branch rows")

# ------------------------------------------------------------
# 5. Build knowledge sentences
# ------------------------------------------------------------
knowledge_docs = []
for row in data_rows:
    branch = row["Branch"]
    for yr, stats in row.items():
        if yr == "Branch":
            continue
        parts = [f"In the academic year {yr}, for {branch}"]
        if stats["Total Students"]:
            parts.append(f"total students were {stats['Total Students']}")
        if stats["Students Placed"]:
            parts.append(f"students placed were {stats['Students Placed']}")
        if stats["Average Salary (LPA)"]:
            parts.append(f"average salary was {stats['Average Salary (LPA)']} LPA")
        if stats["Highest Salary (LPA)"]:
            parts.append(f"highest salary was {stats['Highest Salary (LPA)']} LPA")
        sentence = ", ".join(parts) + "."
        knowledge_docs.append(sentence)

# ------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------
json_path = "placement_stats_knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved {len(knowledge_docs)} entries to {json_path}")

txt_path = "placement_stats_knowledge.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for doc in knowledge_docs:
        f.write(doc + "\n")
print(f"📄 Text version saved to {txt_path}")

# Show a few samples
if knowledge_docs:
    print("\n🔍 Sample sentences:")
    for i, s in enumerate(knowledge_docs[:3], 1):
        print(f"{i}. {s}")