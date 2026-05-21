import pandas as pd
import json

# ------------------------------------------------------------
# 1. Read the already‑cleaned CSV
# ------------------------------------------------------------
CSV_FILE = "./knowledge/cutoff_cleaned.csv"   # <-- change if needed

df = pd.read_csv(CSV_FILE)

# ------------------------------------------------------------
# 2. Map course abbreviations to full names
# ------------------------------------------------------------
course_full = {
    "Mech": "Mechanical Engineering",
    "CSE": "Computer Science & Engineering",
    "RAI": "Robotics and Artificial Intelligence",
    "ELEC": "Electrical Engineering",
    "CS IOT": "Computer Science (IoT & Cyber Security)",
    "Aero": "Aeronautical Engineering",
    "AIDS": "AI & Data Science",
    "CIVIL": "Civil Engineering",
    "Food": "Food Technology"
}

# ------------------------------------------------------------
# 3. Identify category columns (everything after Course and Group)
# ------------------------------------------------------------
# The header looks like: Course, Group, VJ_Merit No, VJ_Merit Marks, NT-1_Merit No, ...
# We'll group them into pairs: (category_name, merit_no_col, merit_marks_col)
category_pairs = []
cols = list(df.columns)
for i in range(2, len(cols), 2):
    if i+1 < len(cols):
        col_no = cols[i]
        col_marks = cols[i+1]
        # Extract category name from column name, e.g., "VJ_Merit No" → "VJ"
        cat = col_no.split("_")[0]   # simple split on underscore
        category_pairs.append((cat, col_no, col_marks))

print(f"Found {len(category_pairs)} category pairs: {[c[0] for c in category_pairs]}")

# ------------------------------------------------------------
# 4. Build sentences, separated by General / Ladies
# ------------------------------------------------------------
general_entries = []
ladies_entries = []

for _, row in df.iterrows():
    course_abbr = row["Course"].strip()
    group_raw = str(row["Group"]).strip()

    # Determine group: "G(General)" or "L(Ledies)"
    group = "General" if "G" in group_raw.upper() else "Ladies" if "L" in group_raw.upper() else "Unknown"
    course_name = course_full.get(course_abbr, course_abbr)

    # Process each category pair
    for cat, col_no, col_marks in category_pairs:
        merit_no = str(row[col_no]).strip() if pd.notna(row[col_no]) else ""
        merit_marks = str(row[col_marks]).strip() if pd.notna(row[col_marks]) else ""

        # Keep "Data not found" as is, empty as "not available"
        if merit_no == "" or merit_no == "-":
            merit_no = "not available"
        if merit_marks == "" or merit_marks == "-":
            merit_marks = "not available"

        sentence = (f"For {group} category in {course_name}, under {cat} category, "
                    f"the cut‑off merit number is {merit_no} and the merit marks are {merit_marks}.")

        if group == "General":
            general_entries.append(sentence)
        elif group == "Ladies":
            ladies_entries.append(sentence)

print(f"General entries: {len(general_entries)}")
print(f"Ladies entries: {len(ladies_entries)}")

# ------------------------------------------------------------
# 5. Save structured JSON
# ------------------------------------------------------------
output = {
    "General": general_entries,
    "Ladies": ladies_entries
}

json_path = "cutoff_knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved structured cutoffs to {json_path}")

# Show samples
if general_entries:
    print("\n🔍 Sample General:")
    for s in general_entries[:2]:
        print(f"• {s}")
if ladies_entries:
    print("\n🔍 Sample Ladies:")
    for s in ladies_entries[:2]:
        print(f"• {s}")