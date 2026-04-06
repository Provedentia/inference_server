from flask import Flask, request, jsonify
import time

app = Flask(__name__)

INFERENCE_MS = 1.0  # match nexus --inference-us 1000

@app.route('/infer', methods=['POST'])
def infer():
    time.sleep(INFERENCE_MS / 1000)
    return jsonify({"result": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081, threaded=True)
