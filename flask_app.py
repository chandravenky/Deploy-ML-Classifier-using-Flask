# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:32:10 2020

@author: vchan
"""

import numpy as np 
import pickle
import pandas as pd
from flask import Flask, request


#Change working directory - only for debugging purposes
#import os
#os.chdir(r'C:\Users\vchan\Documents\Deployed model')

app=Flask(__name__)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#Add a decorator before the function
@app.route('/')
def welcome():
    return "Risk Classification App"

#---------------Based on values giving-------------
@app.route('/prediction')
def preditions():
    age = request.args.get('age')
    marital_status = request.args.get('marital_status')
    income = request.args.get('income')
    
    #Generate prediction
    prediction = classifier.predict([[age, marital_status, income]])

    return "The ML app classifies the prediction as {}".format(prediction[0])

@app.route('/prediction_csv', methods = ["POST"])
def preditions_csv():
    df_test = pd.read_csv(request.files.get("file"))
    
    #Generate prediction
    prediction = classifier.predict(df_test)

    return str(list(prediction))


#This runs the app
if __name__ == "__main__":
    app.run()
    
    