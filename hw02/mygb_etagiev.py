#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'emin.tagiev@phystech.edu'
# Параметрами с которыми вы хотите обучать деревья
#TREE_PARAMS_DICT = {'max_depth': 5}
TREE_PARAMS_DICT = {'max_depth': 8,
                    'min_samples_split':6,
                    'min_samples_leaf': 2,
                   'random_state': 42}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.036


class SimpleGB(BaseEstimator):
    #Need to define sigmoid function for making predictions
    def sigma(self, z):
        z = z.reshape([z.shape[0], 1])
        z[z > 100] = 100
        z[z < -100] = -100
        return 1./(1 + np.exp(-z))
    
    #Gradient of log_loss function
    def log_loss_grad(self, y, p):
        y = y.reshape([y.shape[0], 1])
        p = p.reshape([p.shape[0], 1])
        p[p < 1e-5] = 1e-5
        p[p > 1 - 1e-5] = 1 - 1e-5
        return (p-y)/p/(1-p)
    
    #Let's also try ada boost
    def ada_boos(self, y, p):
        return 

    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        self.means_grad = []
        #curr_pred = self.base_algo.predict(X_data).reshape([y_data.shape[0], 1])
        curr_pred = np.mean(y_data) * np.ones([y_data.shape[0], 1])
        for iter_num in range(self.iters):
            # Нужно посчитать градиент функции потерь
            # TODO
            grad = self.log_loss_grad(y_data, self.sigma(curr_pred))
            #print(-grad)
            self.means_grad.append(grad.mean(axis=0))
            # Нужно обучить DecisionTreeRegressor предсказывать антиградиент
            # Не забудьте про self.tree_params_dict
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, -grad) # TODO
            self.estimators.append(algo)
            # Обновите предсказания в каждой точке
            # TODO
            #print("Predict: ")
            #print(algo.predict(X_data).reshape([X_data.shape[0], 1]))
            curr_pred += self.tau*(algo.predict(X_data).reshape([X_data.shape[0], 1]))
        #print(self.means_grad)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > 0.
