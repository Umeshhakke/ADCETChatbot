import pandas as pd
import json
import re

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
EXCEL_FILE = "./data/FY CUT OFF  Placement data, Fee, Hostel 2025 - 26.xlsx"

# ------------------------------------------------------------
# 1. Find the correct sheet (the one with "First Year" text)
# ------------------------------------------------------------
xls = pd.ExcelFile(EXCEL_FILE)
target_sheet = None
for sheet in xls.sheet_names:
    df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
    for idx, row in df_test.iterrows():
        for cell in row:
            if isinstance(cell, str) and "first year" in cell.lower():
                target_sheet = sheet
                break
        if target_sheet:
            break
    if target_sheet:
        break

if target_sheet is None:
    # Fallback: pick the first sheet that has any text resembling "Document"
    for sheet in xls.sheet_names:
        df_test = pd.read_excel(EXCEL_FILE, sheet_name=sheet, header=None, dtype=str)
        for cell in df_test.iloc[:, 0].dropna():
            if isinstance(cell, str) and "document" in cell.lower():
                target_sheet = sheet
                break
        if target_sheet:
            break

if target_sheet is None:
    raise ValueError("Could not find any sheet with document tables. Available sheets: " + str(xls.sheet_names))

print(f"📌 Using sheet: {target_sheet}")
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=target_sheet, header=None, dtype=str)

# ------------------------------------------------------------
# 2. Find the starting row of each table
# ------------------------------------------------------------
def find_title_row(raw_df, keyword):
    for idx, row in raw_df.iterrows():
        for cell in row:
            if isinstance(cell, str) and keyword.lower() in cell.lower():
                return idx
    return None

first_year_start = find_title_row(df_raw, "first year")
dsy_start = find_title_row(df_raw, "dsy")

print(f"First Year table starts at row: {first_year_start}")
print(f"DSY table starts at row: {dsy_start}")

if first_year_start is None and dsy_start is None:
    raise ValueError("No tables found.")

# ------------------------------------------------------------
# 3. Helper: extract one table given its title row
# ------------------------------------------------------------
def extract_table(raw_df, start_row):
    """
    Finds the category header row (the first row after start_row
    that contains text like 'OPEN / EBC') and then collects all
    data rows until a blank row or next title.
    Returns a DataFrame with columns: 'Document', 'cat1', 'cat2', ...
    """
    if start_row is None:
        return pd.DataFrame()

    # Look for the category header row within the next 3 rows
    cat_row = None
    for i in range(start_row + 1, min(start_row + 4, len(raw_df))):
        row = raw_df.iloc[i]
        # Check if any cell contains "OPEN" (typical of category row)
        for c in range(1, len(row)):  # skip first column (document name)
            val = str(row.iloc[c]).strip() if pd.notna(row.iloc[c]) else ""
            if "open" in val.lower() or "sc" in val.lower():
                cat_row = i
                break
        if cat_row is not None:
            break

    if cat_row is None:
        print(f"   ⚠️ No category header found after row {start_row}")
        return pd.DataFrame()

    # Identify category columns: any column >=1 that has a non‑empty string
    cat_cols = []
    cat_names = []
    for c in range(1, raw_df.shape[1]):
        val = str(raw_df.iloc[cat_row, c]).strip() if pd.notna(raw_df.iloc[cat_row, c]) else ""
        if val:   # this column has a category name
            cat_cols.append(c)
            cat_names.append(val)

    if not cat_cols:
        print("   ⚠️ No category columns found")
        return pd.DataFrame()

    print(f"   Category columns: {cat_cols} → {cat_names}")

    # Collect data rows
    data_rows = []
    for i in range(cat_row + 1, len(raw_df)):
        row = raw_df.iloc[i]
        # Stop if the whole row is empty
        if row.isna().all():
            break
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        # Stop if we hit the next section title
        if re.search(r"(?i)first year|dsy|documents required", first_cell):
            break
        doc_name = first_cell
        values = []
        for c in cat_cols:
            v = str(row.iloc[c]).strip() if pd.notna(row.iloc[c]) else ""
            values.append(v)
        data_rows.append([doc_name] + values)

    if not data_rows:
        return pd.DataFrame()

    df = pd.DataFrame(data_rows, columns=['Document'] + cat_names)
    # Merge cells: empty document name means same as above
    df['Document'] = df['Document'].replace('', pd.NA).fillna(method='ffill')
    df = df.dropna(subset=['Document']).reset_index(drop=True)
    return df

# ------------------------------------------------------------
# 4. Extract both tables
# ------------------------------------------------------------
df_fy = extract_table(df_raw, first_year_start)
df_dsy = extract_table(df_raw, dsy_start)

print(f"First Year rows extracted: {len(df_fy)}")
print(f"DSY rows extracted: {len(df_dsy)}")

# ------------------------------------------------------------
# 5. Generate knowledge sentences
# ------------------------------------------------------------
knowledge_docs = []

def process_table(df, label):
    if df.empty:
        return
    for _, row in df.iterrows():
        doc_name = row['Document'].strip()
        if not doc_name:
            continue
        for col in df.columns[1:]:  # category columns
            category = col.strip()
            val = str(row[col]).strip().lower()
            if val == 'yes':
                sentence = (f"For {label} admission under {category} category, "
                            f"{doc_name} is required.")
                knowledge_docs.append(sentence)
            elif val == 'if applicable':
                sentence = (f"For {label} admission under {category} category, "
                            f"{doc_name} may be required (if applicable).")
                knowledge_docs.append(sentence)

process_table(df_fy, "First Year (B.Tech / BBA / BCA)")
process_table(df_dsy, "DSY (Direct Second Year / Diploma)")

# ------------------------------------------------------------
# 6. Save results
# ------------------------------------------------------------
json_path = "documents_knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
print(f"✅ Saved {len(knowledge_docs)} entries to {json_path}")

# Also a text version
txt_path = "documents_knowledge.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for doc in knowledge_docs:
        f.write(doc + "\n")
print(f"📄 Text version saved to {txt_path}")

# Show a few examples
if knowledge_docs:
    print("\n🔍 Sample sentences:")
    for i, s in enumerate(knowledge_docs[:5], 1):
        print(f"{i}. {s}")
else:
    print("\n⚠️ No sentences generated. Please check the debug output above.")