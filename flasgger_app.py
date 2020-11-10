# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:32:10 2020

@author: vchan
"""

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

#Change working directory - only for debugging purposes
#import os
#os.chdir(r'C:\Users\vchan\Desktop\git\Flassger deployment')

app=Flask(__name__)

#Initiate Swagger app
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#Add a decorator before the function
@app.route('/')
def welcome():
    return "Risk Classification App"

#---------------Create wrapper and function-------------
@app.route('/prediction')
def preditions():
    
    """Credit Risk Modelling 
    Classification model to predict type of risk
    ---
    parameters:  
      - name: age
        in: query
        type: number
        required: true
      - name: marital_status
        in: query
        type: number
        required: true
      - name: income
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    age = request.args.get('age')
    marital_status = request.args.get('marital_status')
    income = request.args.get('income')
    
    #Generate prediction
    prediction = classifier.predict([[age, marital_status, income]])

    return "The ML app classifies the prediction as {}".format(prediction[0])

@app.route('/prediction_csv', methods = ["POST"])
def preditions_csv():
    
    """Credit Risk Modelling 
    Classification model to predict type of risk
    ---
    parameters:  
      - name: file
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    df_test = pd.read_csv(request.files.get("file"))
    
    #Generate prediction
    prediction = classifier.predict(df_test)

    return str(list(prediction))


#This runs the app
if __name__ == "__main__":
    app.run()
    
    