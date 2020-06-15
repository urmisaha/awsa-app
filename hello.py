from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == "__main__":
    app.run("10.129.2.170", "5000")


