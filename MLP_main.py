# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:24:56 2021

@author: AM4
"""
import pandas as pd
import numpy as np
from neural import MLP

df = pd.read_csv('data.csv')

# перемешиваем строки
df = df.iloc[np.random.permutation(len(df))]

# возьмем первые 100 строк, 4-й столбец 
y = df.iloc[0:100, 4].values
# так как ответы у нас строки - нужно перейти к численным значениям
y = np.where(y == "Iris-setosa", 1, 0).reshape(-1,1)

# возьмем два признака
X = df.iloc[0:100, [0, 2]].values

inputSize = X.shape[1] # количество входных сигналов равно количеству признаков задачи 
hiddenSizes = 10 # число нейронов скрытого слоя 
outputSize = 1 if len(y.shape) else y.shape[1] # количество выходных сигналов

iterations = 50
learning_rate = 0.1

net = MLP(inputSize, outputSize, learning_rate, hiddenSizes)

# обучаем сеть по алгоритму стохастического градиентного спуска
for i in range(iterations):
    for xi, yi in zip(X, y):###изменил подачу файла с всего сразу на одну строку x и один y (обновление после каждогопримера)
        xi = xi.reshape(1, -1)###преобразование reshape в матрицу 1*N для матричных операций
        yi = yi.reshape(1, -1)###приводим к формату 1x1 для корректного градиента
        net.train(xi, yi)###обновление весов после каждого прохода а не один раз в эпоху

    if i % 10 == 0:
        print("На итерации: " + str(i) + ' || ' +
              "Средняя ошибка: " + str(np.mean(np.square(y - net.predict(X)))))

# считаем ошибку на обучающей выборке
pr = net.predict(X)
print(sum(abs(y-(pr>0.5))))
