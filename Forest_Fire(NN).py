# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:56:54 2020

@author: VIMAL PM
"""
#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

#loading the dataset using pd.read_csv()
forest=pd.read_csv("D:/DATA SCIENCE/ASSIGNMENT/Work done/Neural Network/forestfires.csv")
#getting the columns names
forest.columns
#Index(['month', 'day', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
   #    'rain', 'area', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
   #    'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
   #   'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
   #    'monthoct', 'monthsep', 'size_category']

#converting categorical to numerical using label encoder()   
from sklearn import preprocessing
Le=preprocessing.LabelEncoder()
Forest=forest
Forest["month"]=Le.fit_transform(forest["month"])
Forest["day"]=Le.fit_transform(forest["day"])

Forest["size_category"]=Le.fit_transform(forest["size_category"])
#checking the missing values
Forest.isnull().sum() #no missing values found
#splitting the input and output variables seperately 
predictors=Forest.iloc[:,:31]
predictors=predictors.drop(["area"],axis=1)
target=Forest.iloc[:,10]
#building the model
Forest_model=Sequential()
Forest_model.add(Dense(90,input_dim=30,activation="relu"))
Forest_model.add(Dense(60,activation="relu"))
Forest_model.add(Dense(20,activation="relu"))
Forest_model.add(Dense(1,kernel_initializer="normal"))
Forest_model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mse"])
#fitting the model
Forest_model.fit(predictors,target,epochs=10)
pred_train=Forest_model.predict(predictors)
Forest["pred_train"]=Forest_model.predict(predictors)
pred_train=pd.Series([i[0] for i in pred_train])
#getting the rmse
rmse_value=np.sqrt(np.mean(pred_train-target)**2)
plt.plot(pred_train,target,"bo")
#getting the correlative coefficent
np.corrcoef(pred_train,target)
#array([[1.        , 0.09166415],
   #    [0.09166415, 1.        ]])
