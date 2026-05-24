from flask import Flask, request, jsonify, render_template
import requests
from rag_chatbot import answer_query

app = Flask(__name__)

# ================= WHATSAPP CONFIG =================

ACCESS_TOKEN = ""
PHONE_NUMBER_ID = ""
VERIFY_TOKEN = "adcet_verify_token"

WHATSAPP_URL = f"https://graph.facebook.com/v25.0/{PHONE_NUMBER_ID}/messages"

# ================= WEBSITE =================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")

    bot_reply = answer_query(user_message)

    return jsonify({
        "reply": bot_reply
    })


# ================= WHATSAPP WEBHOOK =================

@app.route("/webhook", methods=["GET", "POST"])
def webhook():

    # Verification
    if request.method == "GET":
        token = request.args.get("hub.verify_token")
        challenge = request.args.get("hub.challenge")

        if token == VERIFY_TOKEN:
            return challenge

        return "Verification failed", 403

    # Incoming Messages
    if request.method == "POST":
        data = request.json

        try:
            entry = data["entry"][0]
            changes = entry["changes"][0]
            value = changes["value"]

            if "messages" in value:

                message = value["messages"][0]
                phone_number = message["from"]

                if "text" in message:
                    user_text = message["text"]["body"]

                    print("User:", user_text)

                    bot_reply = answer_query(user_text)

                    send_whatsapp_message(phone_number, bot_reply)

        except Exception as e:
            print("Webhook Error:", e)

        return "ok", 200


# ================= SEND WHATSAPP MESSAGE =================

def send_whatsapp_message(to, text):

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": text
        }
    }

    response = requests.post(
        WHATSAPP_URL,
        headers=headers,
        json=payload
    )

    print(response.text)


# ================= RUN SERVER =================

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
