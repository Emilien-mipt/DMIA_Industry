# coding=utf-8
import numpy as np
from sklearn.base import BaseEstimator


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'emin.tagiev@phystech.edu'

LR_PARAMS_DICT = {
    'C': 10.,
    'random_state': 777,
    'iters': 1000,
    'batch_size': 1000,
    'step': 0.01
}


class MyLogisticRegression(BaseEstimator):
    def __init__(self, C, random_state, iters, batch_size, step):
        self.C = C
        self.random_state = random_state
        self.iters = iters
        self.batch_size = batch_size
        self.step = step
    
    #Gradient of log_loss function
    def log_loss_grad(self, y, p):
        y = y.reshape([y.shape[0], 1])
        p = p.reshape([p.shape[0], 1])
        p[p < 1e-5] = 1e-5
        p[p > 1 - 1e-5] = 1 - 1e-5
        return (p-y)

    # будем пользоваться этой функцией для подсчёта <w, x>
    def __predict(self, X):
        return np.dot(X, self.w) + self.w0

    # sklearn нужно, чтобы predict возвращал классы, поэтому оборачиваем наш __predict в это
    def predict(self, X):
        res = self.__predict(X)
        res[res > 0] = 1
        res[res < 0] = 0
        return res

    #Need to define sigmoid function for making predictions
    def predict_proba(self, X):
        a = self.__predict(X)
        return 1/(1+np.exp(-a))

    # производная регуляризатора
    def der_reg(self):
        # TODO
        return (2/self.C)*self.w

    # будем считать стохастический градиент не на одном элементе, а сразу на пачке (чтобы было эффективнее)
    def der_loss(self, x, y):
        # x.shape == (batch_size, features)
        # y.shape == (batch_size,)

        # считаем производную по каждой координате на каждом объекте
        # TODO
        ders_w = x.T@(self.predict_proba(x) - y)

        der_w0 = np.sum(self.predict_proba(x) - y)
        # для масштаба возвращаем средний градиент по пачке
        # TODO
        return ders_w/x.shape[0], der_w0/x.shape[0]

    def fit(self, X_train, y_train):
        # RandomState для воспроизводитмости
        random_gen = np.random.RandomState(self.random_state)
        
        # получаем размерности матрицы
        size, dim = X_train.shape
        
        # случайная начальная инициализация
        self.w = random_gen.rand(dim)
        self.w0 = random_gen.randn()

        for _ in range(self.iters):  
            # берём случайный набор элементов
            rand_indices = random_gen.choice(size, self.batch_size)
            # исходные метки классов это 0/1
            x = X_train[rand_indices]
            y = y_train[rand_indices]

            # считаем производные
            der_w, der_w0 = self.der_loss(x, y)
            der_w += self.der_reg()
            # обновляемся по антиградиенту
            self.w -= der_w * self.step
            self.w0 -= der_w0 * self.step

        # метод fit для sklearn должен возвращать self
        return self