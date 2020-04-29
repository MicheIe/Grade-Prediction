import pandas as pd
dataset = pd.read_excel('/Users/apple/Documents/Computing/Course Work/Final Data/FinalData.xlsx')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values
import numpy as np
from sklearn.preprocessing import Imputer
GapFill = Imputer(missing_values = np.nan, axis = 1)
GapFil = GapFill.fit(X[:,4:-1])
X[:,4:-1] = GapFill.transform(X[:,4:-1])
def KS3(i):
    switcher={
            '7A':6,
            '7B':5,
            '7C':4,
            '6A':3,
            '6B':2,
            '6C':1
            }
    X[ix,iy] = switcher.get(i)
for iy in range(0,2):
    for ix in range(0,299):
        KS3(X[ix,iy])
def ABILITYGROUP(o):
    switcher={
            'AB140+':5,
            'AB130+':4,
            'AB120+':3,
            'AB110+':2,
            'AB100+':1
            }
    X[ox,2] = switcher.get(o)
for ox in range(0,299):
    ABILITYGROUP(X[ox,2])
def FINALGRADE(p):
    switcher={
            'A*':4,
            'A':3,
            'B':2,
            'C':1
            }
    X[px,-1] = switcher.get(p)
for px in range(0,299):
    FINALGRADE(X[px,-1])
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest=tts(X,Y,test_size=0.2)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)
import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
ytrain = scaler.fit_transform(ytrain[:,np.newaxis])
ytest = scaler.transform(ytest[:,np.newaxis])
scalerfile = 'scalery.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
model = Sequential()
model.add(Dense(units=32,kernel_initializer='uniform',activation='relu',input_dim=23))
model.add(Dropout(p=0.1))
model.add(Dense(units=32,kernel_initializer='uniform',activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='linear'))
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mean_absolute_error'])
model.fit(xtrain,ytrain,nb_epoch=1000)
print(model.evaluate((xtest),(ytest),batch_size=32))
y_pred=model.predict(xtest)
Inverseypred=np.round(scaler.inverse_transform(y_pred))
Inverseytest=np.round(scaler.inverse_transform(ytest))
combined=np.column_stack((Inverseytest,Inverseypred)).T
model_json = model.to_json()
with open('model.json','w')as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
print("Saved model to disk")

