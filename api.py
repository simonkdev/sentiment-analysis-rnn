from main_rnn import RNN

from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

rnn = RNN()
rnn.load_trained_state()
rnn.calculate_accuracy()

@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json
    text = data.get('text', '')
    result = rnn.classify_sentiment(text)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5000)