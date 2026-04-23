# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import Perceptron


df = pd.read_csv('data.csv')

df = df.iloc[np.random.permutation(len(df))]
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[0:100, [0, 2]].values


inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # задаем число нейронов скрытого (А) слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов равно количеству классов задачи


NN = Perceptron(inputSize, hidden1=10, hidden2=5, outputSize=1)###два слоя 1-10 нейронов 2-5 нейронов


NN.train(X, y, n_iter=5, eta = 0.01)

y = df.iloc[:, 4].values
y = np.where(y == "Iris-setosa", 1, -1)
X = df.iloc[:, [0, 2]].values
out, h1, h2 = NN.predict(X)###возврат 3-х значений(итог+1слой+2слой)


errors = np.sum(out.reshape(-1) != y)###сделал вывод ошибок по количеству
print("Ошибок:", errors)

