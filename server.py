from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

from main_rnn import RNN

app = Flask(__name__)
CORS(app)

rnn = RNN()
rnn.load_trained_state()


@app.route("/", methods=["GET"])
def index():
    return (Path(__file__).resolve().parent / "public" / "index.html").read_text(encoding="utf-8")


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
