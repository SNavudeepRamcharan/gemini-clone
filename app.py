import os
import base64
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

SYSTEM_RULE = (
    "You are Next AI, an advanced language model developed by S.Navudeep Ram Charan. "
    "You remember images and files shared previously in the same conversation chat thread. "
    "When a user asks to solve a matrix problem, provide a clear, efficient explanation and final answer."
)

# Active session data dictionary store supporting image/file caching blocks
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
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set."}), 500
    if not user_message and not uploaded_file:
        return jsonify({"error": "No prompt or file received"}), 400

    # Initialize chat history thread metadata array stores if missing
    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "File Analysis")
        title = title_source if len(title_source) <= 30 else title_source[:27] + "..."
        chat_database[session_id] = {
            "title": title,
            "messages": [],
            "cached_files": []  # NEW: Tracks persistent image memory bounds for this session
        }

    # 1. Process and permanently cache file inputs inside the active database thread structure
    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            
            # Save file structural parts directly into session data cache bounds
            chat_database[session_id]["cached_files"].append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to process file component: {str(file_err)}"}), 400

    # 2. Append current user message string trace inside logs database mapping references
    if user_message:
        chat_database[session_id]["messages"].append({"role": "user", "text": user_message})
    else:
        chat_database[session_id]["messages"].append({"role": "user", "text": "📂 Sent an image file attachment"})

    # 3. Compile full historical structural records array payload block
    contents = []
    
    # Bundle previous historical conversation blocks safely
    for msg in chat_database[session_id]["messages"][:-1]:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })
        
    # 4. Construct current prompt containing text trace + ALL historical file data vectors from this session
    current_request_parts = []
    
    # Inject cached files if they exist so Next AI maintains visual continuity
    if chat_database[session_id]["cached_files"]:
        current_request_parts.extend(chat_database[session_id]["cached_files"])
        
    # Inject current text prompt string parameters
    current_request_parts.append({"text": user_message if user_message else "Analyze the attached context."})
    
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
        response = requests.post(url, json=payload, timeout=25)
        response_data = response.json()
        
        if 'candidates' not in response_data:
            return jsonify({"error": "AI processing timeout or structural delivery breakdown. Try resending."}), 400
            
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": f"Transmission breakdown: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)