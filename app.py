import os
import base64
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# System rule tightly optimized to keep generation fast and avoid server timeout crashes
SYSTEM_RULE = (
    "You are Next AI, an advanced language model developed by S.Navudeep Ram Charan. "
    "You remember images and files shared previously in the same conversation chat thread. "
    "CRITICAL: When a user uploads a large matrix (like a 10x10 matrix), do NOT print out the intermediate "
    "matrices for every single row reduction step, as it will hit a timeout error. "
    "Instead, provide a highly concise answer: state the key row operation strategy briefly, "
    "and provide the final Reduced Row Echelon Form (RREF) and the inverse matrix right away."
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
        return jsonify({"error": "GEMINI_API_KEY environment variable is not set."}), 500
    if not user_message and not uploaded_file:
        return jsonify({"error": "No prompt or file received"}), 400

    if session_id not in chat_database:
        title_source = user_message if user_message else (uploaded_file.filename if uploaded_file else "Matrix Analysis")
        title = title_source if len(title_source) <= 30 else title_source[:27] + "..."
        chat_database[session_id] = {
            "title": title,
            "messages": [],
            "cached_files": []
        }

    if uploaded_file and uploaded_file.filename != '':
        try:
            file_bytes = uploaded_file.read()
            base64_data = base64.b64encode(file_bytes).decode('utf-8')
            mime_type = uploaded_file.content_type
            
            chat_database[session_id]["cached_files"].append({
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64_data
                }
            })
        except Exception as file_err:
            return jsonify({"error": f"Failed to process file component: {str(file_err)}"}), 400

    if user_message:
        chat_database[session_id]["messages"].append({"role": "user", "text": user_message})
    else:
        chat_database[session_id]["messages"].append({"role": "user", "text": "📂 Sent an image file attachment"})

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
        # Extended timeout close to Render's absolute max limit to prevent early connection dropping
        response = requests.post(url, json=payload, timeout=29)
        response_data = response.json()
        
        if 'candidates' not in response_data:
            return jsonify({"error": "The problem processing limit was reached. Please try splitting the request details."}), 400
            
        ai_response = response_data['candidates'][0]['content']['parts'][0]['text']
        chat_database[session_id]["messages"].append({"role": "model", "text": ai_response})
        return jsonify({"response": ai_response})

    except requests.exceptions.Timeout:
        return jsonify({"error": "The calculation is extremely heavy and timed out. Try asking me for the final answer directly without showing steps!"}), 504
    except Exception as e:
        return jsonify({"error": f"Transmission breakdown: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)