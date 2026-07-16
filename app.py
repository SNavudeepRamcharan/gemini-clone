import os
import base64
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# System rule modified to ensure the model responds efficiently to large matrices
SYSTEM_RULE = (
    "You are Next AI, an advanced language model developed by S.Navudeep Ram Charan. "
    "When a user asks to solve an exceptionally large matrix (like a 10x10 matrix), "
    "do not write out all 100 rows for every single step all at once, as it will cause a timeout. "
    "Instead, explain the initial row operations clearly, show the first few pivotal reductions, "
    "and state the final Reduced Row Echelon Form (RREF) matrix cleanly so the connection stays stable."
)

chat_database = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/history', methods=['POST'])
def get_specific_history():
    data = request.json or {}
    device_sessions = data.get("session_ids", [])
    summary = []
    for sid in device_sessions:
        if sid in chat_database:
            summary.append({"session_id": sid, "title": chat_database[sid]["title"]})
    return jsonify(summary)

@app.route('/api/load_session', methods=['POST'])
def load_session_history():
    data = request.json or {}
    session_id = data.get("session_id")
    if session_id in chat_database:
        return jsonify({"messages": chat_database[session_id]["messages"]})
    return jsonify({"messages": []})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_message = request.form.get("message", "")
    session_id = request.form.get("session_id", "default")
    uploaded_file = request.files.get("file")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set on Render."}), 500
    if not user_message and not uploaded_file:
        return jsonify({"error": "No prompt or file received."}), 400

    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "Matrix Analysis")
        title = title_source if len(title_source) <= 30 else title_source[:27] + "..."
        chat_database[session_id] = {"title": title, "messages": []}

    current_request_parts = []

    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            
            current_request_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to parse image file data: {str(file_err)}"}), 400

    if user_message:
        current_request_parts.append({"text": user_message})
        chat_database[session_id]["messages"].append({"role": "user", "text": user_message})
    else:
        # If the user only sent an image without text, append a placeholder prompt
        current_request_parts.append({"text": "Analyze this problem image and solve it efficiently."})
        chat_database[session_id]["messages"].append({"role": "user", "text": "📂 Sent an image for analysis"})

    contents = []
    for msg in chat_database[session_id]["messages"][:-1]:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })
        
    contents.append({
        "role": "user",
        "parts": current_request_parts
    })

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_RULE}]}
    }
    
    try:
        # Added a 25-second timeout constraint to prevent Render from abruptly cutting the stream
        response = requests.post(url, json=payload, timeout=25)
        response_data = response.json()
        
        if 'candidates' not in response_data:
            return jsonify({"error": "The AI engine is busy or throttled. Please try again in a moment."}), 400
            
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except requests.exceptions.Timeout:
        return jsonify({"error": "The matrix calculation is too heavy and timed out. Try asking to break it down step-by-step!"}), 504
    except Exception as e:
        return jsonify({"error": f"Server processing bottleneck: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)