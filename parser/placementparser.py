import pandas as pd
import json
import re

# -----------------------------------------------
# CONFIGURATION
# -----------------------------------------------
EXCEL_FILE = "./data/placement.xlsx"

# -----------------------------------------------
# 1. Find the correct sheet
# -----------------------------------------------
xls = pd.ExcelFile(EXCEL_FILE)
target_sheet = None

for sheet in xls.sheet_names:
    df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
    for idx, row in df_test.iterrows():
        for cell in row:
            if isinstance(cell, str) and "company visited list" in cell.lower():
                target_sheet = sheet
                break
        if target_sheet:
            break
    if target_sheet:
        break

if target_sheet is None:
    # Fallback: look for a sheet with "TCS" or "KPIT" in first column
    for sheet in xls.sheet_names:
        df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
        if any("TCS" in str(cell) for cell in df_test.iloc[:,0].dropna()):
            target_sheet = sheet
            break

if target_sheet is None:
    raise ValueError("Could not find placement data. Sheets: " + str(xls.sheet_names))

print(f"📌 Using sheet: {target_sheet}")
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=target_sheet, header=None, dtype=str)

# -----------------------------------------------
# 2. Locate all year-title rows
# -----------------------------------------------
title_rows = []
for idx, row in df_raw.iterrows():
    for cell in row:
        if isinstance(cell, str) and "company visited list" in cell.lower():
            title_rows.append(idx)
            break

if not title_rows:
    raise ValueError("No 'Company Visited List' rows found.")
print(f"Found {len(title_rows)} year-title rows at indices: {title_rows}")

# -----------------------------------------------
# 3. Extract data for each year
# -----------------------------------------------
all_data = []   # list of (year_label, DataFrame)

for i, title_idx in enumerate(title_rows):
    # Extract year string from the title cell
    title_text = ""
    for cell in df_raw.iloc[title_idx]:
        if isinstance(cell, str) and "company visited list" in cell.lower():
            title_text = cell
            break
    year_match = re.search(r"(\d{4}-\d{2,4})", title_text)
    year_label = year_match.group(1) if year_match else f"Year_{i+1}"

    # Find the header row (first non-empty row after title)
    header_idx = None
    for j in range(title_idx + 1, min(title_idx + 4, len(df_raw))):
        if not df_raw.iloc[j].isna().all():
            header_idx = j
            break
    if header_idx is None:
        print(f"   ⚠️ Could not find header row for {year_label}")
        continue

    # Determine where to stop: next title row or end of DataFrame
    next_title = title_rows[i+1] if i+1 < len(title_rows) else len(df_raw)

    # Collect data rows – SKIP blank rows, DO NOT break on them
    data_rows = []
    for j in range(header_idx + 1, next_title):
        row = df_raw.iloc[j]
        # Skip completely empty rows (they might appear before the actual data)
        if row.isna().all():
            continue

        company = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        industry = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        branches = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""

        # Skip header repeats or rows with no company name
        if company.lower() in ["company name", ""]:
            continue

        data_rows.append([company, industry, branches])

    if data_rows:
        df = pd.DataFrame(data_rows, columns=['Company', 'Industry', 'Branches'])
        all_data.append((year_label, df))
        print(f"   ✅ {year_label}: {len(df)} companies extracted")
    else:
        print(f"   ❌ {year_label}: no data rows found")

# -----------------------------------------------
# 4. Build knowledge sentences
# -----------------------------------------------
knowledge_docs = []

branch_full = {
    "CSE": "Computer Science & Engineering",
    "Ele": "Electrical Engineering",
    "Mech": "Mechanical Engineering",
    "Civil": "Civil Engineering",
    "Aero": "Aeronautical Engineering",
    "Food": "Food Technology",
    "AIDS": "AI & Data Science",
    "IOT": "IoT & Cyber Security",
    "IT": "Information Technology",
    "All Branch": "all branches",
}

for year_label, df in all_data:
    for _, row in df.iterrows():
        company = row['Company'].strip()
        industry = row['Industry'].strip()
        branches_raw = row['Branches'].strip()

        # Replace abbreviations with full names (word boundaries)
        branches_clean = branches_raw
        for abbr, full in branch_full.items():
            branches_clean = re.sub(r'\b' + re.escape(abbr) + r'\b', full, branches_clean)
        branches_clean = re.sub(r'\s+', ' ', branches_clean).strip()

        industry_part = f" ({industry})" if industry else ""
        sentence = (f"In the academic year {year_label}, {company}{industry_part} visited the campus "
                    f"and recruited from the following branches: {branches_clean}.")
        knowledge_docs.append(sentence)

# -----------------------------------------------
# 5. Save results
# -----------------------------------------------
json_path = "placements_knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
print(f"\n✅ Saved {len(knowledge_docs)} entries to {json_path}")

txt_path = "placements_knowledge.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for doc in knowledge_docs:
        f.write(doc + "\n")
print(f"📄 Text version saved to {txt_path}")

# Show a few examples
if knowledge_docs:
    print("\n🔍 Sample sentences:")
    for i, s in enumerate(knowledge_docs[:3], 1):
        print(f"{i}. {s}")
else:
    print("\n⚠️ No entries generated. If the script found title rows but no data, please share the debug output.")