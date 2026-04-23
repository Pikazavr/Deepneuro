import numpy as np

class Perceptron:
    def __init__(self, inputSize, hidden1, hidden2, outputSize):###добавил еще один скрытый слой 

        ###Скрытый слой 1
        self.W1 = np.zeros((1 + inputSize, hidden1))
        self.W1[0, :] = np.random.randint(0, 3, size=(hidden1))###матрица весов w1
        self.W1[1:, :] = np.random.randint(-1, 2, size=(inputSize, hidden1))

        ###Скрытый слой 2
        self.W2 = np.zeros((1 + hidden1, hidden2))
        self.W2[0, :] = np.random.randint(0, 3, size=(hidden2))###матрица весов w2
        self.W2[1:, :] = np.random.randint(-1, 2, size=(hidden1, hidden2))

        ###Выходной слой
        self.Wout = np.random.randint(0, 2, size=(1 + hidden2, outputSize)).astype(np.float64)

    def predict(self, Xp):
        ###Слой 1
        h1 = np.where((np.dot(Xp, self.W1[1:, :]) + self.W1[0, :]) >= 0, 1, -1).astype(np.float64)

        ###Слой 2
        h2 = np.where((np.dot(h1, self.W2[1:, :]) + self.W2[0, :]) >= 0, 1, -1).astype(np.float64)

        ###Выход
        out = np.where((np.dot(h2, self.Wout[1:, :]) + self.Wout[0, :]) >= 0, 1, -1).astype(np.float64)

        return out, h1, h2

    def train(self, X, y, n_iter=5, eta=0.01):
        for _ in range(n_iter):
            for xi, target in zip(X, y):
                pr, h1, h2 = self.predict(xi)###добавил второй скрытый слой в обучение h2

                self.Wout[1:] += ((eta * (target - pr)) * h2).reshape(-1, 1)
                self.Wout[0] += eta * (target - pr)

        return self
