import json

# Hard‑coded based on your data – adjust if needed
criteria_sentences = [
    "For Open category Engineering admission, the qualifying criteria is: PCM or PM + any Technical Subject, with a minimum of 134/135 marks.",
    "For Reserved category Engineering admission, the qualifying criteria is: PCM or PM + any Technical Subject, with a minimum of 119/120 marks."
]

with open("qualifying_criteria_knowledge.json", "w", encoding="utf-8") as f:
    json.dump(criteria_sentences, f, indent=2, ensure_ascii=False)

print("✅ Saved qualifying criteria entries.")