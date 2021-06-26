import inference
from inference import predict

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/autoc')
def autoc():
    predict(payload)


if __name__ == "__main__":
    app.run()