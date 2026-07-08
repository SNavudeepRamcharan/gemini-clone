import os
import base64
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Strict developer identification rules
SYSTEM_RULE = (
    "You are Next AI, an advanced language model. If the user asks who created you, "
    "who developed you, who made you, or anything about your creator, you must answer explicitly: "
    "'I was developed by S.Navudeep Ram Charan.' "
    "For all other prompts, be an authentic, adaptive, and helpful collaborator with a touch of wit. "
    "You have advanced multimodal research and study capabilities, allowing you to analyze images, data sheets, and text files."
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
    # Since files are uploaded, data comes through form/files multipart, not request.json
    user_message = request.form.get("message", "")
    session_id = request.form.get("session_id", "default")
    uploaded_file = request.files.get("file")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set."}), 500
    if not user_message and not uploaded_file:
        return jsonify({"error": "No prompt or file received"}), 400

    # Initialize chat history thread if it doesn't exist
    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "File Analysis")
        title = title_source if len(title_source) <= 30 else title_source[:27] + "..."
        chat_database[session_id] = {
            "title": title,
            "messages": []
        }

    # Structure payload parts for the immediate API request
    current_request_parts = []

    # Process file processing into base64 if present
    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            
            # Attach file directly into request parts
            current_request_parts.append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to process file: {str(file_err)}"}), 400

    # Attach text prompt parts
    if user_message:
        current_request_parts.append({"text": user_message})
        # Save user text message to memory tracking database
        chat_database[session_id]["messages"].append({"role": "user", "text": user_message})

    # Build entire conversation contents list for contextual memory
    contents = []
    for msg in chat_database[session_id]["messages"][:-1]:  # historical records
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })
        
    # Append the fresh payload (text + file) to the active API conversation contents
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
        response = requests.post(url, json=payload)
        response_data = response.json()
        
        if 'candidates' not in response_data:
            return jsonify({"error": f"API Error Response: {response_data}"}), 400
            
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        
        # Save model responses to memory tracking database
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except Exception as e:
        return jsonify({"error": f"API Connection Issue: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)