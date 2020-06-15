from flask import Flask, render_template
from predict import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/desc')
def desc():
    return render_template('desc.html')

@app.route('/predict')
def predict(text):
    result = calculate_sentiment(text)
    if result is None:
        abort(404)
    return result

if __name__ == "__main__":
    app.run("10.129.2.170", "5001")

