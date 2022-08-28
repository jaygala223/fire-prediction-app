from flask import Flask, render_template, request
import requests
import pickle
import numpy as np
from sklearn import *


model = pickle.load(open('model.pkl','rb'))

def predict_fire(temp, wind, humidity):

    input = np.array([[temp, wind, humidity]]).astype(np.float64)
    prediction = model.predict_proba(input)
    return prediction[0][1]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    temp = request.form.get('temperature')
    humidity = request.form.get('humidity')
    wind = request.form.get('wind')

    output = predict_fire(temp, wind, humidity)
    
    return render_template('predict.html', output=output)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)