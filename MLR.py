import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data=pd.read_csv("texi.csv")
#print(data.head())

x = data.iloc[:,0:-1]
y = data.iloc[:,-1]
#print(y_train)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

reg=LinearRegression()
reg.fit(x_train,y_train)
pickle.dump(reg,open('texi.pkl','wb'))
model=pickle.load(open('texi.pkl','rb'))

#print(model.predict([[80,1770000,6000,85]]))
