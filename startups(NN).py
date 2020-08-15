# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:48:43 2020

@author: VIMAL PM
"""
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

#loading the dataset using pd.read_csv()
Startups=pd.read_csv("D:/DATA SCIENCE/ASSIGNMENT/Work done/Neural Network/50_startups.csv")
startups=Startups
startups.columns
#Index(['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit'], dtype='object')

from sklearn import preprocessing
Le=preprocessing.LabelEncoder()
startups["State"]=Le.fit_transform(Startups["State"])
startups.isnull().sum()

#spliting the data
predictors=startups.iloc[:,:4]
target=startups.iloc[:,4]

#bulding the model
startup_model=Sequential()
startup_model.add(Dense(80,input_dim=4,activation="relu"))
startup_model.add(Dense(50,activation="relu"))
startup_model.add(Dense(40,activation="relu"))
startup_model.add(Dense(1,kernel_initializer="normal"))
startup_model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mse"])
#fitting my model
startup_model.fit(predictors,target,epochs=10)
pred_train=startup_model.predict(predictors)
startups["pred_train"]=startup_model.predict(predictors)
pred_train=pd.Series(i[0] for i in pred_train)

#getting the rmse
rmse_value=np.sqrt(np.mean(pred_train-target)**2)
plt.plot(pred_train,target,"bo")
#getting the correlative coefficent
np.corrcoef(pred_train,target)
#array([[1.        , 0.80716158],
   #   [0.80716158, 1.        ]]
#There is an higher correlation b/w the predicted result and the target result.   
