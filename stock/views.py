from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View,TemplateView
from django.http import JsonResponse
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import pickle

# Create your views here.
 
class Model(View):
    
    def preprocess(self):
        B_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df=pd.read_csv(os.path.join(B_DIR,"stock_dataset","APPLE.csv"))
        data = df.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
        for i in range(0,len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]
        timestamp = new_data['Date'].to_frame(name='Date')
        new_data.index = new_data.Date
        new_data.drop('Date', axis=1, inplace=True)

        #creating train and test sets
        dataset = new_data.values

        train = dataset[0:1700,:]
        valid = dataset[1700:,:]  

        #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset) 
        x_train, y_train = [], []
        for i in range(60,len(train)):
            x_train.append(scaled_data[i-60:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    
        return x_train,y_train,train,valid,new_data,scaler,timestamp

    def train(self):
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)

        return model    
    


def index(request):
    return render(request,'index.html')

def chart(request):
    return render(request,'charts.html')
    
def findList(request):
    BaseDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = Model()
    x_train,y_train,train,valid,new_data,scaler,timestamp = model.preprocess()
    timestamp_train = timestamp[0:1700]
    timestamp_test = timestamp[1700:]  
    filename = os.path.join(BaseDIR,"stock_dataset","LSTM.sav")
    model=pickle.load(open(filename, 'rb'))

    #predicting 246 values, using past 60 from the train data
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:1700]
    valid = new_data[1700:]
    valid['Predictions'] = closing_price

    data_dict = { "Close" : valid['Close'].tolist(),"Predictions" : valid['Predictions'].tolist(), "Date":timestamp_test['Date'].tolist()}
    return HttpResponse(json.dumps(data_dict), content_type='application/json')
    #return model.get(request)    