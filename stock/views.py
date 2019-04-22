from django.shortcuts import render
from django.http import HttpResponse,HttpResponseRedirect
from django.views.generic import View,TemplateView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import json
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import pickle
from stock.models import Timestamp
from django.views.generic import View,TemplateView
from yahoo_finance import Share
import datetime
import socket
import fix_yahoo_finance as yf 
from stock.forms import UserForm, UserProfileInfoForm
from django.contrib.auth import authenticate,login,logout
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import logging

from keras import backend as K
# Create your views here.
 
 # Get an instance of a logger
logger = logging.getLogger(__name__)

class Model(View):
        BaseDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        def updateCSV(self,folder,csv,company):
                filename = os.path.join(Model.BaseDIR,"stock_dataset",folder,csv)
               
                df = pd.read_csv(filename)
                last_row=df[-1:]
                lst = last_row["Date"].str.split(pat="-")
                lst = lst.tolist() 
                x = datetime.datetime(int(lst[0][0],base=10),int(lst[0][1],base=10),int(lst[0][2],base=10))
                x += datetime.timedelta(days=1)
                now = datetime.datetime.now()
                print("------ X date----")
                print(x)
        
                if(x.date()<now.date()):
                        data = yf.download(company,x.strftime("%Y-%m-%d"),now.strftime("%Y-%m-%d"))
                        
                        with open(filename, 'a') as f:
                                data.to_csv(f, header=False) 

        def preprocess(self,folder,csv):
                #B_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                df=pd.read_csv(os.path.join(Model.BaseDIR,"stock_dataset",folder,csv))
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

        def train(self,x_train,y_train,folder,pickle_file):
                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(units=50))
                model.add(Dense(1))

                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2)

                # filename = './stock_dataset/LSTM.sav'
                filename = os.path.join(Model.BaseDIR,"stock_dataset",folder,pickle_file)
                pickle.dump(model, open(filename, 'wb'))

                result=Timestamp.objects.first()        
                result.timeStamp = datetime.datetime.now()
                result.save()

                return model                      
    

        def train(self,x_train,y_train,folder,pickle_file):
                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(units=50))
                model.add(Dense(1))

                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=2)

                # filename = './stock_dataset/LSTM.sav'
                filename = os.path.join(Model.BaseDIR,"stock_dataset",folder,pickle_file)
                pickle.dump(model, open(filename, 'wb'))

                result=Timestamp.objects.first()        
                result.timeStamp = datetime.datetime.now()
                result.save()

                return model                      
    

def register(request):
    registered=False

    if request.method == "POST":
        #print(json.loads(request.body))
        user_form = UserForm(data=request.POST)
        profile_form = UserProfileInfoForm(data=request.POST)

        if user_form.is_valid() and profile_form.is_valid():
            user = user_form.save()
            user.set_password(user.password)
            user.save()

            profile = profile_form.save(commit=False)
            profile.user = user
            profile.save()

            registered= True
        else: print(user_form.errors, profile_form.errors)
    else:
        user_form = UserForm
        profile_form = UserProfileInfoForm

    return render(request, 'register.html',{
                                                'user_form':user_form,
                                                   'profile_form':profile_form
                                                    })
@login_required
def user_logout(request):
        logout(request)
        return HttpResponseRedirect(reverse('index'))


def user_login(request):

        if(request.method=="POST"):
                username=request.POST.get('username')
                password=request.POST.get('password')

                user = authenticate(username=username, password=password)
                if user:
                        if user.is_active:
                                login(request, user)
                                return HttpResponseRedirect(reverse('index'))
                        else:
                                return HttpResponse("Acoount not active")
                else:
                        return HttpResponse("Invalid Login Details")
        
        return render(request,'login.html',{})




@csrf_exempt
@login_required
def index(request):
    logger.error('Something went wrong!')    
    return render(request,'index.html')

@csrf_exempt
def chart(request):
    return render(request,'charts.html')

@csrf_exempt    
def findList(request):
            
    current_date = datetime.datetime.strptime(str(Timestamp.objects.first()), '%Y-%m-%d %H:%M:%S')
    now = datetime.datetime.now()
    BaseDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    json_object = json.loads(request.body)
    id = json_object["id"]

    folder = None
    csv = None
    pickle_file = None

    if(id == 1):
        folder = "GOGLE"
        csv = "GOGLE.csv"
        pickle_file = "GOGLE.sav"
    elif(id == 2):
        folder = "AAPL"
        csv = "AAPL.csv"
        pickle_file = "AAPL.sav"
    elif(id == 3):
        folder = "AMZN"
        csv = "AMZN.csv"
        pickle_file = "AMZN.sav"
    else: 
        folder = "MS"
        csv = "MS.csv"
        pickle_file = "MS.sav"


    mdl = Model()

    x_train,y_train,train,valid,new_data,scaler,timestamp = mdl.preprocess(folder,csv)
    timestamp_train = timestamp[0:1700]
    timestamp_test = timestamp[1700:] 
        
    model = None    
    if(current_date.date() == now.date()):
            #read pickle
            print("--------------reading pickle----------------------")
            filename = os.path.join(BaseDIR,"stock_dataset",folder,pickle_file)
            model=pickle.load(open(filename, 'rb'))
            
    else:
            print("--------------Updating model----------------------")
            
            mdl.updateCSV("AAPL","AAPL.csv","AAPL")
            model = mdl.train(x_train,y_train,"AAPL","AAPL.sav")

            mdl.updateCSV("AMZN","AMZN.csv","AMZN")
            model = mdl.train(x_train,y_train,"AMZN","AMZN.sav") 

            mdl.updateCSV("MS","MS.csv","MSFT")
            model = mdl.train(x_train,y_train,"MS","MS.sav")            

            mdl.updateCSV("GOGLE","GOGLE.csv","GOOGL")
            model = mdl.train(x_train,y_train,"GOGLE","GOGLE.sav")   

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    K.clear_session()    
    train = new_data[:1700]
    valid = new_data[1700:]
    valid['Predictions'] = closing_price

    data_dict = { "Close" : valid['Close'].tolist(),"Predictions" : valid['Predictions'].tolist(), "Date":timestamp_test['Date'].tolist()}
    return HttpResponse(json.dumps(data_dict), content_type='application/json')