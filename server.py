from flask import Flask, jsonify, request
from flask_cors import CORS

from web_page import INDEX_HTML

app = Flask(__name__)
CORS(app)

_rnn = None


def get_rnn():
    global _rnn
    if _rnn is None:
        from main_rnn import RNN

        _rnn = RNN()
        _rnn.load_trained_state()
    return _rnn


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

    scores = get_rnn().predict_sentiment_scores(text)
    result = "positive" if scores[1] > scores[0] else "negative"
    return jsonify({
        "result": result,
        "scores": {
            "negative": float(scores[0]),
            "positive": float(scores[1]),
        },
    })


if __name__ == "__main__":
    app.run(port=5000)
