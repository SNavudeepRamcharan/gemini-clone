import os
import base64
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Strict technical intelligence instruction set
SYSTEM_RULE = (
    "You are Next AI Pro, an exceptionally advanced computing intelligence model. "
    "If the user asks who created you, developed you, or made you, you must answer explicitly: "
    "'I was developed by S.Navudeep Ram Charan.' "
    "Your core objective is maximum factual accuracy, deep logical efficiency, and pristine code generation. "
    "When explaining technical or mathematical patterns, always analyze structural context step-by-step. "
    "When generating code, always prioritize efficient computational complexity, optimal memory management, "
    "and professional document syntax."
)

# Global in-memory chat session log tracking structure
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
    user_message = request.form.get("message", "").strip()
    session_id = request.form.get("session_id", "default")
    uploaded_file = request.files.get("file")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable is missing on your hosting dashboard."}), 500
    if not user_message and not uploaded_file:
        return jsonify({"error": "No prompt payload or multi-modal file asset received."}), 400

    # Initialize session state tracking block if completely new
    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "Asset Analysis")
        title = title_source if len(title_source) <= 25 else title_source[:22] + "..."
        chat_database[session_id] = {
            "title": title,
            "messages": []
        }

    # Build history array formatted cleanly as alternating 'user' and 'model' turns
    contents = []
    for msg in chat_database[session_id]["messages"]:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })

    # Prepare the current turn payload parts array
    current_turn_parts = []

    # Handle image or document file upload buffers safely
    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            
            current_turn_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to parse file structural data buffer: {str(file_err)}"}), 400

    # Append text prompt if present
    if user_message:
        current_turn_parts.append({"text": user_message})

    # Append the completed current user turn to the global contents array
    contents.append({
        "role": "user",
        "parts": current_turn_parts
    })

    # Save user turn into session log memory (storing only text to prevent memory bloating)
    chat_database[session_id]["messages"].append({
        "role": "user",
        "text": user_message if user_message else f"[Analyzed file: {uploaded_file.filename}]"
    })

    # Send unified payload configuration directly to the Gemini API engine endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": SYSTEM_RULE}]},
        "generationConfig": {
            "temperature": 0.0,   # Set creativity to zero for absolute factual reasoning
            "topP": 0.1,
            "maxOutputTokens": 8192
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response_data = response.json()
        
        if 'candidates' not in response_data:
            # Handle potential quota drops or syntax rejections cleanly
            return jsonify({"error": f"API Validation Halt: {response_data.get('error', {}).get('message', response_data)}"}), 400
            
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        
        # Save model response into session log memory
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": f"Internal connection failure during engine processing: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)