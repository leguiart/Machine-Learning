import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod
from hklearn.model import Model

class _BaseModel(Model):
    def __init__(self, estimator = 'ML', alpha = 1.0):        
        if isinstance(estimator, str) and estimator.lower() in ['ml', 'map']:
            self.estimator = estimator.lower()
        else:
            raise Exception("Not valid estimator option (either ML or MAP)")
        self.alpha = alpha

    def fit(self, X, y):
        if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
            X_np = X.to_numpy()
        else:
            X_np = X
        
        if isinstance(y, pd.core.frame.DataFrame):
            y_np = y.to_numpy()
        else:
            y_np = y
        self.classes = np.unique(y)
        self.class_estimators, self.attribute_estimators = self.estimate(X_np, y_np, self.classes)

    @abstractmethod
    def estimate(self, X, y, classes):
        pass

    @abstractmethod
    def predict(self, X):
        pass
    

class BernoulliNB(_BaseModel):    
    def estimate(self, X, y, classes):
        m = classes.shape[0]
        n = X.shape[1]
        beta = n
        attribute_estimators = np.zeros((m,n))
        class_estimators = np.zeros((m))
        for i, c in enumerate(classes):
            X_c = X[np.where(y == c)]
            if self.estimator == 'ml':
                class_estimators[i] = X_c.shape[0]/X.shape[0]
                attribute_estimators[i, :] = np.count_nonzero(X_c, axis = 0) / X_c.shape[0]
            elif self.estimator == 'map':
                class_estimators[i] = (X_c.shape[0] + self.alpha - 1)/(X.shape[0] + beta + self.alpha - 2)
                attribute_estimators[i, :] = (np.count_nonzero(X_c, axis = 0) + self.alpha - 1) / (X_c.shape[0] + beta + self.alpha - 2)
        return class_estimators, attribute_estimators

    def predict(self, X):
        if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
            X_np = X.to_numpy()
        else:
            X_np = X
        X_np = np.where(X_np > 0.5, 1., 0.)
        pcc = np.zeros((self.classes.shape[0], X_np.shape[0]))
        a0log = (1 - X_np) @ np.log(1 - self.attribute_estimators + 0.000000001).T
        a1log = X_np @ np.log(self.attribute_estimators + 0.000000001).T
        pcc = a0log + a1log + np.log(self.class_estimators)
        return np.argmax(pcc, axis = 1)

class MultinomialNB(_BaseModel):
    def estimate(self, X, y, classes):
        m = classes.shape[0]
        n = X.shape[1]
        beta = n
        attribute_estimators = np.zeros((m,n))
        class_estimators = np.zeros((m))
        for i, c in enumerate(classes):
            X_c = X[np.where(y == c)]
            if self.estimator == 'ml':
                class_estimators[i] = X_c.shape[0]/X.shape[0]
                n_w = X_c.sum(axis = 0)
                attribute_estimators[i, :] = n_w / n_w.sum()
            elif self.estimator == 'map':
                n_w = X_c.sum(axis = 0)
                class_estimators[i] = (X_c.shape[0] + self.alpha - 1)/(X.shape[0] + m * self.alpha - m)
                attribute_estimators[i, :] = (n_w + self.alpha - 1) / (n_w.sum() + n * self.alpha - n)
        return class_estimators, attribute_estimators


    def predict(self, X):
        if isinstance(X, (pd.core.frame.DataFrame, pd.core.series.Series)):
            X_np = X.to_numpy()
        else:
            X_np = X
        pcc = np.zeros((self.classes.shape[0], X_np.shape[0]))
        a1log = X_np @ np.log(self.attribute_estimators + 0.000000001).T
        pcc = a1log + np.log(self.class_estimators)
        return np.argmax(pcc, axis = 1)
