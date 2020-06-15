from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
# from predict import *

app = Flask(__name__)

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        domain = request.form['domain']
        text = request.form['review']
        print(domain)
        print(text)
        # result = calculate_sentiment(text)
        # if result is None:
            # abort(404)
    return render_template('index.html')

@app.route('/desc')
def desc():
    return render_template('desc.html')

if __name__ == "__main__":
    app.run("10.129.2.170", "5001")

