from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from django.http import HttpResponse
from wsgiref.util import FileWrapper
from django.template.response import TemplateResponse
import json


def home(request):
    df = pd.read_csv(
        r"E:\Đại học\Năm III\HK I\Machine Learning\LT\Do an\Destinationdiretory3.csv")
    df = df[:8]
    # data = data.to_html()
    json_records = df.reset_index().to_json(orient='records')
    arr = []
    arr = json.loads(json_records)
    contextt = {'d': arr}
    return render(request, 'index.html', contextt)


def predict(request):
    return render(request, 'predict.html')


def result(request):
    dataset = pd.read_csv(
        r"E:\Đại học\Năm III\HK I\Machine Learning\LT\Do an\Destinationdiretory3.csv")
    X = dataset.drop("diabetes", axis=1)
    y = dataset.diabetes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=150)  # 20 % train
    model_LR = LogisticRegression()
    model_LR.fit(X_train, y_train)
    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    pred_LR = model_LR.predict(
        [[val1, val2, val3, val4, val5, val6, val7, val8]])
    result1 = ""

    if pred_LR == [1]:
        result1 = " Bạn bị mắt bệnh tiểu đường"
    else:
        result1 = " Bạn không mắt bệnh tiểu đường"
    return render(request, 'predict.html', {"result2": result1})
