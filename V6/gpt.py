import requests
import json

API_KEY = "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"   # <-- replace with your key securely
# Try a free model that supports reasoning (DeepSeek-R1)
MODEL = "openai/gpt-oss-120b:free"

def call_openrouter(messages, reasoning=True):
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    if reasoning:
        payload["reasoning"] = {"enabled": True}

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Reasoning Test",
        },
        json=payload,
        timeout=60
    )

    print(f"Status: {resp.status_code}")
    resp_json = resp.json()
    if resp.status_code != 200:
        print("❌ Error response:")
        print(json.dumps(resp_json, indent=2))
        return None

    # Print full response (first 500 chars for readability)
    print("📦 Full response (truncated):")
    print(json.dumps(resp_json, indent=2)[:1000])

    if "choices" not in resp_json:
        print("❌ No 'choices' in response")
        return None

    return resp_json["choices"][0]["message"]

# --- First call ---
print("🔹 First call:")
msg1 = call_openrouter([{"role": "user", "content": "How many r's in 'strawberry'?"}])
if msg1:
    print("Assistant content:", msg1.get("content"))
    print("Reasoning details:", msg1.get("reasoning_details"))
else:
    exit("First call failed")

# --- Second call (with preserved reasoning_details) ---
print("\n🔹 Second call (follow‑up with reasoning):")
messages = [
    {"role": "user", "content": "How many r's in 'strawberry'?"},
    {
        "role": "assistant",
        "content": msg1.get("content"),
        "reasoning_details": msg1.get("reasoning_details")  # pass back as‑is
    },
    {"role": "user", "content": "Are you sure? Think carefully."}
]
msg2 = call_openrouter(messages)
if msg2:
    print("Assistant content:", msg2.get("content"))