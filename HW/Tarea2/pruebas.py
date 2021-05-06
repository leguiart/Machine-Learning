# Standard scientific Python imports
#import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from hklearn.naive_bayes import BernoulliNB, MultinomialNB
import numpy as np
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/gibranfp/CursoAprendizajeAutomatizado/master/data/dep_inf.csv')
# Split data into 50% train and 50% test subsets
# X_train, X_test, y_train, y_test = train_test_split(
#     data, digits.target, test_size=0.5, shuffle=False)

X_nuevos = np.array([[1, 0, 0, 1, 1, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 0]])
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
bnblm = MultinomialNB(estimator='ML', alpha = 2.0)
bnblm.fit(X, y)

print(bnblm.predict(X_nuevos))