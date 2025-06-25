import os
from flask import Flask, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# sart flask app
app = Flask(__name__)
CORS(app)

# home page
@app.route("/", methods=['GET'])
def index():
    prediction_url = os.environ["PREDICTION_URL"]
    print(prediction_url)
    return render_template('index.html', prediction_url=prediction_url)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
