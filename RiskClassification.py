# -*- coding: utf-8 -*-
"""
@author: vchan
"""

#------------Exercise 1-------------

from sklearn.neighbors import KNeighborsClassifier
import numpy
import pandas
import pickle

risk_df = pandas.read_csv(r"C:\Users\vchan\Documents\Deployed model\train.csv")

#Subset the required columns
X = risk_df.iloc[:,2:5]
y = risk_df['risk']

#Standardize the columns
from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

#Split the train and test rows
X_std_train = X_std[1:,:]
X_std_test = X_std[:1,:]
y_train= y[1:,]

#Build model
from sklearn.neighbors import KNeighborsClassifier
knn2 = KNeighborsClassifier(n_neighbors=2, p=2).fit(X_std_train,y_train)

#Predict
#predict(X_std_test)

#Generate pickle file using serialization

pickle_out = open("classifier.pkl","wb")
pickle.dump(knn2, pickle_out)
pickle_out.close()









