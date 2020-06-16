from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from models import *
import os
from predict_ontoweighted import *
from predict_unweighted import *

app = Flask(__name__)

@app.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        domain = request.form['domain']
        text = request.form['review']
        onto, unwt = calculate_sentiment(domain, text)

        if onto is None or unwt is None:
            abort(404)
        
        if int(onto) == 1:
            onto = 'positive'
        else:
            onto = 'negative'

        if int(unwt) == 1:
            unwt = 'positive'
        else:
            unwt = 'negative'

        result = {'onto': onto, 'unwt': unwt}
        return render_template('index.html', result=result)
    return render_template('index.html')

@app.route('/desc')
def desc():
    return render_template('desc.html')

def calculate_sentiment(domain, text):
    print(domain)
    print(text)
    predict_onto = onto_prediction(domain, text)
    predict_unwt = unwt_prediction(domain, text)
    print("model run.. results obtained")
    return predict_onto, predict_unwt

if __name__ == "__main__":
    app.run("10.129.2.170", "5002")

