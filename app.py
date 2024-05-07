import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def feature():
    return render_template('features2.html')

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/form', methods=['GET','POST'])
def form():
    return render_template('predict2.html')

prediction_text = []
input_data = []

@app.route('/predict', methods=['GET','POST'])
def predict():

    float_features = [float(x) for x in request.form.values()]
    mean_value = [49870.1312,50.3584,0.5087,0.4968,50.2763,50225.4861,5.5101,5.5217,2005.4885,0.4991,0.4999,5033.1039,5028.0106,553.1212,0.503,4.9946]
    std_dev_value = [28774.37535,28.81669637,0.499949302,0.500014761,28.88917127,29006.6758,2.872024172,2.856666793,9.308089589,0.500024192,0.500024992,2876.729545,2894.33221,262.0501699,0.500016001,3.176409891]
    reverse_scaled_features = []
    for i in range(len(float_features)):
        reverse_scaled_features.append((float_features[i] - mean_value[i])/std_dev_value[i])
    features = [np.array(reverse_scaled_features)]
    prediction = model.predict(features)


    # return render_template('predict2.html','result.html',input_data = float_features,prediction_text = prediction)
    return render_template('result.html',input_data = float_features,prediction_text = prediction)

@app.route('/result', methods=['GET','POST'])
def result():
    return render_template('result.html',input_data,prediction_text)

if __name__ == '__main__':
    app.run(debug=True)