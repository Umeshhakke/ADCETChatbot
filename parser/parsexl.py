import pandas as pd
import json
import re

# ------------------------------------------------
# 1. Load raw sheet (no predefined header)
# ------------------------------------------------
EXCEL_FILE = "./data/FY CUT OFF  Placement data, Fee, Hostel 2025 - 26.xlsx"
SHEET_NAME = "Sheet1"

df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, header=None, dtype=str)

# ------------------------------------------------
# 2. Helpers to detect sections & extract tables
# ------------------------------------------------
def find_section_start(raw_df, keyword):
    """Return row index where the section header containing 'keyword' is found."""
    for idx, row in raw_df.iterrows():
        for cell in row:
            if isinstance(cell, str) and keyword.lower() in cell.lower():
                return idx
    return None

def extract_section_table(raw_df, start_row):
    """
    From start_row+1, collect rows until a completely empty row
    or a new 'Program Offered' header, or end of DataFrame.
    First non‑empty row after start_row is used as column headers.
    """
    # find column header row
    col_row = start_row + 1
    while col_row < len(raw_df) and raw_df.iloc[col_row].isna().all():
        col_row += 1
    if col_row >= len(raw_df):
        return pd.DataFrame()

    columns = [str(c).strip() if isinstance(c, str) else f"col{i}" for i, c in enumerate(raw_df.iloc[col_row])]

    data_rows = []
    for i in range(col_row + 1, len(raw_df)):
        row = raw_df.iloc[i]
        if row.isna().all():
            break
        first_cell = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        if "program offered" in first_cell.lower():
            break
        data_rows.append(row.values)

    if not data_rows:
        return pd.DataFrame()

    df_section = pd.DataFrame(data_rows, columns=columns)
    df_section = df_section.dropna(axis=1, how='all').reset_index(drop=True)
    return df_section

# ------------------------------------------------
# 3. Identify section start rows
# ------------------------------------------------
bt_start = find_section_start(df_raw, "b.tech")
ug_start = find_section_start(df_raw, "bba")        # may be None if only B.Tech & M.Tech present
mt_start = find_section_start(df_raw, "m.tech")

if bt_start is None or mt_start is None:
    raise ValueError("Could not locate B.Tech or M.Tech sections. Check sheet layout.")

# ------------------------------------------------
# 4. Extract each section as a clean DataFrame
# ------------------------------------------------
df_btech = extract_section_table(df_raw, bt_start)
df_ug = extract_section_table(df_raw, ug_start) if ug_start else pd.DataFrame()
df_mtech = extract_section_table(df_raw, mt_start)

def clean_columns(df):
    """Rename columns to standard names we can work with."""
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'program' in col_lower and 'code' in col_lower:
            col_map[col] = 'Program Code'
        elif 'program' in col_lower:
            col_map[col] = 'Programs'
        elif 'intake' in col_lower:
            col_map[col] = 'Sanctioned Intake'
        elif 'year' in col_lower:
            col_map[col] = 'Year of Starting'
        elif 'sr' in col_lower or 'no' in col_lower:
            col_map[col] = 'Sr.No'
        else:
            col_map[col] = col
    return df.rename(columns=col_map)

df_btech = clean_columns(df_btech)
if not df_ug.empty:
    df_ug = clean_columns(df_ug)
df_mtech = clean_columns(df_mtech)

# Drop serial number column if it exists
for df_section in [df_btech, df_ug, df_mtech]:
    if 'Sr.No' in df_section.columns:
        df_section.drop('Sr.No', axis=1, inplace=True)

# ------------------------------------------------
# 5. Build knowledge sentences
# ------------------------------------------------
knowledge_docs = []

def safe_year(val):
    try:
        return int(float(str(val).strip()))
    except:
        return None

def safe_code(val):
    if pd.isna(val) or str(val).strip() == '':
        return "Not available"
    val_str = str(val).strip()
    # if it's a number like '628361210', keep it as string
    return val_str

def safe_intake(val):
    try:
        return int(float(str(val).strip()))
    except:
        return str(val).strip()

# B.Tech
for _, row in df_btech.iterrows():
    prog = str(row['Programs']).strip()
    code = safe_code(row.get('Program Code', 'Not available'))
    intake = safe_intake(row['Sanctioned Intake'])
    year = safe_year(row['Year of Starting'])
    if year is None or prog is None:
        continue
    tense = "started" if year < 2026 else "starting"
    doc = (f"The institute offers a B.Tech in {prog} "
           f"(Program Code: {code}) with a sanctioned intake of {intake} students, "
           f"{tense} in {year}.")
    knowledge_docs.append(doc)

# Undergraduate (BBA / BCA)
if not df_ug.empty:
    for _, row in df_ug.iterrows():
        prog = str(row['Programs']).strip()
        code = safe_code(row.get('Program Code', 'Not available'))
        intake = safe_intake(row['Sanctioned Intake'])
        year = safe_year(row['Year of Starting'])
        if year is None or prog is None:
            continue
        tense = "started" if year < 2026 else "starting"
        doc = (f"The institute offers a {prog} "
               f"(Program Code: {code}) with a sanctioned intake of {intake} students, "
               f"{tense} in {year}.")
        knowledge_docs.append(doc)

# M.Tech
for _, row in df_mtech.iterrows():
    prog = str(row['Programs']).strip()
    intake = safe_intake(row['Sanctioned Intake'])
    year = safe_year(row['Year of Starting'])
    if year is None or prog is None:
        continue
    doc = (f"The institute offers an M.Tech in {prog} "
           f"with a sanctioned intake of {intake} students, "
           f"starting in {year}.")
    knowledge_docs.append(doc)

# ------------------------------------------------
# 6. Save to JSON (and optional TXT)
# ------------------------------------------------
json_path = "knowledge_base.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(knowledge_docs, f, indent=2, ensure_ascii=False)
print(f"✅ Saved {len(knowledge_docs)} documents to {json_path}")

# Also a plain text version for easy reading
txt_path = "knowledge_base.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for doc in knowledge_docs:
        f.write(doc + "\n")
print(f"📄 Also saved human‑readable version to {txt_path}")

# ------------------------------------------------
# 7. (Optional) Embed and store in vector DB
# ------------------------------------------------
# Uncomment below when you have OPENAI_API_KEY set and libraries installed
#
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
#
# embeddings = OpenAIEmbeddings()
# vectordb = Chroma.from_texts(knowledge_docs, embeddings, persist_directory="./chroma_db")
# vectordb.persist()
# print("✅ Vector database stored in ./chroma_db")

# Verify a few documents
print("\n🔍 Sample entries:")
for i, doc in enumerate(knowledge_docs[:3], 1):
    print(f"{i}. {doc}")