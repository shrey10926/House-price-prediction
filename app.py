# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods = ['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('index.html')

    if request.method == 'POST':
        crim = request.form['crim']
        nox = request.form['nox']
        rm = request.form['rm']
        age = request.form['age']
        dis = request.form['dis']
        tax = request.form['tax']
        b = request.form['b']
        lstat = request.form['lstat']
        
        input_variables = pd.DataFrame([[crim, nox, rm, age, dis, tax, b, lstat]],
                                       columns=['CRIM', 'NOX', 'RM', 'AGE', 'DIS',
                                                'TAX', 'B', 'LSTAT'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return render_template('index.html', original_input={'CRIM': crim, 'NOX': nox, 'RM': rm, 'AGE': age, 'DIS': dis, 'TAX': tax, 'B': b, 'LSTAT': lstat},
                                     result=predictions)



'''
@app.route('/predict', methods = ['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_test = 'Price is ()'.format(output))


@app.route('/predict_api', methods = ['POST'])
def predict_api():
    
    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])
    
    output = prediction[0]
    return jsonify(output)
'''

if __name__ == '__main__':
    app.run(debug = True)





