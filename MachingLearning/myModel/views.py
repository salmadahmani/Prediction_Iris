from django.shortcuts import render,redirect
import pickle

import numpy as np






# Create your views here.
def Home(request):
    return render(request, 'home.html')
def predict(request):
    model = pickle.load(open('IRISdmodel.sav', 'rb'))

    data1 = request.GET['a']
    data2 = request.GET['b']
    data3 = request.GET['c']
    data4 = request.GET['d']
    arr = np.array([[data1, data2, data3, data4]])
    result = model.predict(arr)

    return render(request, "predict.html", {'result': result})
