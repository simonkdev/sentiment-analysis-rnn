import main_rnn as rnn

from flask import Flask, request, jsonify
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json
    text = data.get('text', '')
    result = rnn.forward_pass(text)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(port=5000)