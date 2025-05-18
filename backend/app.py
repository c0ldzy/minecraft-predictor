import os
from flask import Flask, request, jsonify
from model.model import predict_image

app = Flask(__name__, static_folder=os.path.join(os.pardir, 'frontend'))

@app.route("/")
def index():
    return app.send_static_file('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify(error='No file'), 400
    img = request.files['file']
    return jsonify(predict_image(img))

if __name__ == "__main__":
    app.run(debug=True)
