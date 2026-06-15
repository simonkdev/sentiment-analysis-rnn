from flask import Flask, jsonify, request
from flask_cors import CORS

from main_rnn import RNN

INDEX_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Text Processor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
        textarea, button { width: 100%; margin: 10px 0; }
        #result { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
    </style>
</head>
<body>
    <h1>Text Processor</h1>
    <textarea id="inputText" rows="4" placeholder="Enter text here..."></textarea>
    <button onclick="processText()">Process</button>
    <div id="result"></div>

    <script>
        async function processText() {
            const text = document.getElementById('inputText').value;
            const response = await fetch('/api/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });
            const data = await response.json();
            document.getElementById('result').textContent = data.result || data.error;
        }
    </script>
</body>
</html>
"""

app = Flask(__name__)
CORS(app)

rnn = RNN()
rnn.load_trained_state()


@app.route("/", methods=["GET"])
def index():
    return INDEX_HTML


@app.route("/favicon.ico", methods=["GET"])
@app.route("/favicon.png", methods=["GET"])
def favicon():
    return "", 204


@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text is required."}), 400

    result = rnn.classify_sentiment(text)
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(port=5000)
