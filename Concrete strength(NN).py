# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:13:10 2020

@author: VIMAL PM
"""
#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

#loading the datasets using pd.read_csv()
concrete=pd.read_csv("D:\\DATA SCIENCE\\Data sets\\concrete.csv")
#getting the first 5 obsevations of my dataset
concrete.head()
#getting the columns names
concrete.columns
#Index(['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg',
    #   'fineagg', 'age', 'strength']

#checking the missing values
concrete.isnull().sum()#no missing values
#Next I'm going to split my input and output variables seperately as predictors and target
predictors=concrete.iloc[:,0:8]
target=concrete.iloc[:,8]
#building my model
concrete_model=Sequential()

concrete_model.add(Dense(50,input_dim=8,activation="relu"))
concrete_model.add(Dense(40,activation="relu"))
concrete_model.add(Dense(20,activation="relu"))
concrete_model.add(Dense(1,kernel_initializer="normal"))
concrete_model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mse"])

#fitting model to the dataset
concrete_model.fit(predictors,target,epochs=10)

pred_train=concrete_model.predict(predictors)
concrete["pred_train"]=concrete_model.predict(predictors)
pred_train=pd.Series([i[0] for i in pred_train])
#geting the rmse
rmse_value=np.sqrt(np.mean(pred_train-target)**2)
plt.plot(pred_train,target,"bo")
#getting the correlation coefficent
np.corrcoef(pred_train,target)
#array([[1.        , 0.83094139],

#      [0.83094139, 1.        ]])
#83% correlation which is higher
