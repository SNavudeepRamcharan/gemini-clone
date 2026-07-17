import os
import base64
from flask import Flask, render_template, request, Response, jsonify
import requests
import json

app = Flask(__name__)

SYSTEM_RULE = (
    "You are Next AI, a premium language model developed by S.Navudeep Ram Charan. "
    "You possess advanced linear algebra, code architecture, and multimodal research capabilities. "
    "You maintain flawless continuity and visual memory of images/files sent in earlier turns of the conversation. "
    "Format mathematical systems or matrices inside clean standard LaTeX array/pmatrix syntax blocks. "
    "Be direct, outstandingly accurate, and highly analytical."
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
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server."}), 500

    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "New Thread")
        title = title_source if len(title_source) <= 30 else title_source[:27] + "..."
        chat_database[session_id] = {"title": title, "messages": [], "cached_files": []}

    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            chat_database[session_id]["cached_files"].append({
                "inlineData": {"mimeType": mime_type, "data": base64_data}
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to process file attachment: {str(file_err)}"}), 400

    if user_message:
        chat_database[session_id]["messages"].append({"role": "user", "text": user_message})
    else:
        chat_database[session_id]["messages"].append({"role": "user", "text": "📂 Sent an image context loop"})

    contents = []
    for msg in chat_database[session_id]["messages"][:-1]:
        contents.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["text"]}]
        })
        
    current_request_parts = []
    if chat_database[session_id]["cached_files"]:
        current_request_parts.extend(chat_database[session_id]["cached_files"])
    current_request_parts.append({"text": user_message if user_message else "Analyze the attached context."})
    contents.append({"role": "user", "parts": current_request_parts})

    def generate_tokens():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?key={api_key}"
        payload = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": SYSTEM_RULE}]}
        }
        
        full_ai_response = ""
        try:
            response = requests.post(url, json=payload, stream=True, timeout=60)
            
            for chunk in response.iter_lines():
                if chunk:
                    chunk_text = chunk.decode('utf-8').strip()
                    
                    # Clean up streaming JSON array syntax wrappers if present
                    if chunk_text.startswith(","):
                        chunk_text = chunk_text[1:].strip()
                    if chunk_text.startswith("["):
                        chunk_text = chunk_text[1:].strip()
                    if chunk_text.endswith("]"):
                        chunk_text = chunk_text[:-1].strip()
                        
                    if not chunk_text:
                        continue
                        
                    try:
                        data = json.loads(chunk_text)
                        token = data['candidates'][0]['content']['parts'][0]['text']
                        full_ai_response += token
                        yield token
                    except:
                        continue
                        
            chat_database[session_id]["messages"].append({"role": "model", "text": full_ai_response})
            
        except Exception as e:
            yield f" [Streaming bottleneck encountered: {str(e)}] "

    return Response(generate_tokens(), mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)