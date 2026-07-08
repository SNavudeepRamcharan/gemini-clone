import os
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Strict developer identification rules
SYSTEM_RULE = (
    "You are Next AI, an advanced language model. If the user asks who created you, "
    "who developed you, who made you, or anything about your creator, you must answer explicitly: "
    "'I was developed by S.Navudeep Ram Charan.' "
    "For all other prompts, be an authentic, adaptive, and helpful collaborator with a touch of wit."
)

# Global database structure: { session_id: { "title": "...", "messages": [...] } }
chat_database = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/history', methods=['POST'])
def get_specific_history():
    """Returns only the titles for the specific session IDs that belong to the user's device."""
    data = request.json or {}
    device_sessions = data.get("session_ids", [])
    
    summary = []
    for sid in device_sessions:
        if sid in chat_database:
            summary.append({"session_id": sid, "title": chat_database[sid]["title"]})
    return jsonify(summary)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message")
    session_id = data.get("session_id", "default")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set."}), 500
    if not user_message:
        return jsonify({"error": "No prompt received"}), 400

    if session_id not in chat_database:
        title = user_message if len(user_message) <= 30 else user_message[:27] + "..."
        chat_database[session_id] = {
            "title": title,
            "messages": []
        }

    chat_database[session_id]["messages"].append({"role": "user", "text": user_message})

    contents = []
    for msg in chat_database[session_id]["messages"]:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_RULE}]}
    }
    
    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": f"API Connection Issue: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)